
from __future__ import annotations

import pprint, copy, weakref
import typing as T
from collections import defaultdict
from dataclasses import dataclass

from wplib import inheritance, dictlib, log
from wplib.serial import serialise, deserialise, Serialisable, SerialAdaptor
from wplib.object import DeepVisitor, Adaptor, Proxy, ProxyData, ProxyMeta, VisitObjectData, DeepVisitOp, Visitable, VisitAdaptor
from wplib.constant import SEQ_TYPES, MAP_TYPES, STR_TYPES, LITERAL_TYPES, IMMUTABLE_TYPES
from wplib.sentinel import Sentinel
from wpdex.base import WpDex, DexPathable


class WpDexProxyData(ProxyData):
	parent : weakref.ref
	wpDex : WpDex
	deltaStartState : T.Any
	deltaCallDepth : int

class ReactLogic(Adaptor):
	"""silo off the active parts of react to avoid jamming in
	even more base classes, it's getting mental"""
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	forTypes = (object, )


class WpDexProxyMeta(ProxyMeta):
	"""manage top-level call logic -
	if you call WpDexProxy(obj) raw in client code,
	wrap the full hierarchy in proxies and return the root

	we specify some flags to prevent infinite loops on meta call
	parent=None - pass the parent proxy, if known
	isRoot=None - None starts the hierarchy wrapping, True is the root, False is a child
	"""

	def __call__(cls:type[WpDexProxy], *args, **kwargs):
		log("WPMETA call", cls, args, type(args[0]), kwargs)
		parent = kwargs.pop("parent", None)
		isRoot = kwargs.pop("isRoot", None)

		# called at top level, wrap hierarchy
		# if isRoot is None and parent is None:
			#return cls.wrap(args[0], **kwargs)
		return super().__call__(*args, **kwargs)



class WpDexProxy(Proxy, metaclass=WpDexProxyMeta):
	"""replace a hieararchy of objects in-place:
	copy the proxy idea of subclassing for each type.
	deserialise-copy the root item, and for each lookup type
	item, create a new type inheriting from that and this class

	unsure if this class should also inherit from WpDex itself,
	don't see a reason why not

	inheriting from Proxy just to get the class generation

	ADMIT DEFEAT on watching vanilla structures - replace at source with this

	even more critical not to pollute namespace, since these are the actual
	live objects
	use WpDex definitions on which methods to watch

	IMMUTABLE TYPES -
	we're not adding functionality to edit strings in-place, so we delegate
	hash to the immutable object as normal - all the proxy does is watch it
	and provide a hook for rx. the source stays the same
	"""
	_classProxyCache : dict[type, dict[type, type]] = defaultdict(dict) # { class : { class cache } }

	_proxyAttrs = {"_reactData" : {}}
	_generated = False
	_objIdProxyCache : dict[int, WpDexProxy] = {}

	def __init__(self, obj, proxyData: WpDexProxyData=None,
	             wpDex:WpDex=None, parentDex:WpDex=None,
	             **kwargs):
		"""create a WpDex object for this proxy, add
		weak sets for children and parent"""
		log("WP init", obj, type(obj), type(self), self._proxyParentCls, self._proxySuperCls)
		self._proxySuperCls.__init__(self, obj, proxyData, **kwargs)


		self._proxyData["deltaStartState"] = None
		self._proxyData["deltaCallDepth"] = 0

		# wpdex set up
		if wpDex is not None: # pre-built dex passed in
			self._proxyData["wpDex"] = wpDex
		else:
			self._proxyData["wpDex"] = WpDex(obj)
			# link parent dex if given
		if parentDex is not None:
			self.dex().parent = parentDex

		self.updateProxy()

	def dex(self)->WpDex:
		return self._proxyData["wpDex"]

	@classmethod
	def wrap(cls, obj:T.Any, op:DeepProxyOp=None, **kwargs):
		"""return the given structure recursively wrapped in proxies
		there's no good way to do this in a single pass -
		we go bottom-up to create the proxy hierarchy, then top-down
		to set the links between parents and children

		"""
		v = DeepVisitor()
		op = op or DeepProxyOp(proxyCls=cls, )
		r = v.dispatchPass(obj, passParams=v.VisitPassParams(
			topDown=False, depthFirst=True,
			transformVisitedObjects=True,
			visitFn=op.visit, visitRoot=True,
			rootObj=obj
		))
		return r

	def _openDelta(self):
		"""open a delta for this object -
		if mutating functions are reentrant, only open one delta"""
		if self._proxyData["deltaCallDepth"] == 0:
			self._proxyData["deltaStartState"] = copy.deepcopy(self._proxyData["target"])
			self.dex().prepForDeltas()
		self._proxyData["deltaCallDepth"] += 1

	def _emitDelta(self):
		"""emit a delta for this object"""
		self._proxyData["deltaCallDepth"] -= 1
		if self._proxyData["deltaCallDepth"] == 0:
			# delta = DeltaAtom(self._proxyData["deltaStartState"], self._proxyData["target"])
			self._proxyData["deltaStartState"] = None
			# send out delta event
			# TODO: make an actual delta event
			#log("send event path", self.dex().path)

			deltaMap = self.dex().gatherDeltas()
			if deltaMap:
				event = {"type":"deltas", "deltas":deltaMap,
				         "path" : self.dex().path}
				self.dex().sendEvent(event)

			# event = {"type":"delta", "delta":None,
			#                       "path" : self.dex().path
			#                       }
			# log("event destinations", self.dex()._allEventDestinations(
			# 	event, "main"
			# ), self.dex()._nextEventDestinations(event, "main"))




	def _beforeProxyCall(self, methodName:str,
	                     methodArgs:tuple, methodKwargs:dict,
	                     targetInstance:object
	                     ) ->tuple[T.Callable, tuple, dict, object]:
		"""open a delta if mutating method called"""
		#log(f"before proxy call {methodName}, {methodArgs, methodKwargs}", vars=0)
		fn, args, kwargs, targetInstance = super()._beforeProxyCall(methodName, methodArgs, methodKwargs, targetInstance)

		# if not self.dex().childIdDexMap:
		# 	self.dex().updateChildren()
		#
		# # check if method will mutate data - need to open a delta
		# if methodName in self.dex().mutatingMethodNames:
		# 	self._openDelta()

		return fn, args, kwargs, targetInstance

	def _afterProxyCall(self, methodName:str,
	                    method:T.Callable,
	                     methodArgs:tuple, methodKwargs:dict,
	                    targetInstance: object,
	                    callResult:object,
	                    ) ->object:
		"""return result wrapped in a wpDex proxy, if it appears in
		main dex children

		gather deltas, THEN refresh children, THEN emit deltas
		"""
		#log(f"after proxy call {methodName}, {methodArgs, methodKwargs}", vars=0)
		callResult = super()._afterProxyCall(methodName, method, methodArgs, methodKwargs, targetInstance, callResult)
		toReturn = callResult
		# if methodName in { "__call__", "traverse"}:
		# 	log(methodName, " result", callResult, type(callResult))
		# 	log(" ", self._objIdProxyCache)
		# 	log(" ", self._existingProxy(callResult))


		# if mutating method called, rebuild WpDex children
		#log("method", methodName, "in", methodName in self.dex().mutatingMethodNames)
		if methodName in self.dex().mutatingMethodNames:
			"""TO UPDATE CHILDREN -
			need to keep THIS object intact - 
			run op on root's TARGET, 
			DO NOT wrap that object,
			REBASE root (this proxy) on to the result
			
			but then we invalidate all references to any branch, regardless of it changing
			i am in deep pain
			i just want to make films man
			"""
			log("AFTER", methodName, "update children mutating method" )

			# ensure every bit of the structure is still wrapped in a proxy
			#self._proxyData["target"] = self.wrap(self._proxyData["target"])
			#self._proxyData["target"] = WpDexProxy(self._proxyData["target"])
			self.updateProxy()

			#self.dex().updateChildren()


		# if mutating method called, finallyemit delta
		if methodName in self.dex().mutatingMethodNames:
			self._emitDelta()

		# consider instance methods that "return self"
		# check if a proxy already exists for that object and return it
		checkExisting = self._existingProxy(toReturn)
		if checkExisting is not None:
			toReturn = checkExisting
		return toReturn

	def _onProxyCallException(self,
	                          methodName: str,
	                          method:T.Callable,
	                          methodArgs: tuple, methodKwargs: dict,
	                          targetInstance: object,
	                          exception:BaseException) ->(None, object):
		"""ensure we still emit / clear deltas if an exception is raised"""
		log(f"on proxy call exception {methodName}, {methodArgs, methodKwargs}")
		if methodName in self.dex().mutatingMethodNames:
			self._emitDelta()
		raise exception

	# def __hash__(self):
	# 	log("hash proxy called")
	# 	return hash(self._proxyData["target"])

	def updateProxy(self):
		""" we can't keep regenerating new proxy objects or no references will
		ever be stable
		we can't just update objects directly because immutable types exist

		compare CHILD objects, find any that are NOT PROXIES
		create proxies of them,
		update the TARGET OF THIS PROXY to the newObj(), formed by THOSE PROXIES.

		before update, save each proxy and its value against its path on the root
		"""
		#log("update proxy", self)
		adaptor = VisitAdaptor.adaptorForObject(self._proxyData["target"])
		childObjects = list(adaptor.childObjects(self._proxyData["target"], {}))
		# returns list of 3-tuples (index, value, childType)
		needsUpdate = False
		for i, t in enumerate(childObjects):
			if isinstance(t[1], WpDexProxy):
				t[1].updateProxy()
			else:
				needsUpdate = True

				# ADD IN PATH LOOKUP HERE to check if proxy is already known
				#log("wrap child", t[1])
				proxy = WpDexProxy(t[1], isRoot=False)
				#log("result proxy", proxy, type(proxy))
				# if proxy is not None:
				# 	proxy.updateProxy()
				childObjects[i] = (t[0], proxy, t[2])
		if needsUpdate:
			#log("   updating", self, childObjects)
			#log([type(i[1]) for i in childObjects])
			newObj = adaptor.newObj(self._proxyData["target"], childObjects, {})
			#log("final childObjects", adaptor.childObjects(newObj, {}))
			#log([type(i[1]) for i in adaptor.childObjects(newObj, {})])
			self._setProxyTarget(self, newObj)
			#self._proxyData["target"] = newObj

		for i in adaptor.childObjects(self._proxyData["target"], {}):
			if i[1] is None:
				continue
			# if isinstance(i[1], WpDexProxy):
			#
			# 	# maybe we don't need to proxy primitive types?
			#
			# 	i[1].updateProxy()

	# @class
	# def updateRecursive(self):
	#
	# 	adaptor = VisitAdaptor.adaptorForObject(self._proxyData["target"])
	# 	childObjects = list(adaptor.childObjects(self._proxyData["target"], {}))
	# 	for i, t in enumerate(childObjects):
	# 		if not isinstance(t[1], WpDexProxy):
	# 			needsUpdate = True
	#
	# 			# ADD IN PATH LOOKUP HERE to check if proxy is already known
	# 			childObjects[i] = (t[0], WpDexProxy(t[1], isRoot=False), t[2])


@dataclass
class DeepProxyOp(DeepVisitOp):
	proxyCls : type
	rootObj : T.Any
	transformRoot : bool = True

	def visit(self,
	          obj:T.Any,
              visitor:DeepVisitor,
              visitObjectData:VisitObjectData,
              #visitPassParams:VisitPassParams,
              ) ->T.Any:
		"""Transform to apply to each object during deserialisation.
		"""
		log("visit", obj, type(obj))
		log("data", visitObjectData)
		isRoot = False
		if obj is visitObjectData["visitPassParams"].rootObj:
			log("visit root" )
			isRoot = True
		return self.proxyCls(obj, proxyData={}, parent=None,
		                     isRoot=isRoot)


# class ProxyRecursive(Proxy):
# 	pass

if __name__ == '__main__':
	s = [[3]]
	# v = DeepVisitor()
	# op = DeepProxyOp()
	# r = v.dispatchPass(s, passParams=v.VisitPassParams(
	# 	topDown=False, depthFirst=True, transformVisitedObjects=True,
	# 	visitFn=op.visit
	# ))
	r = WpDexProxy(s)
	print(r, type(r))
	print(r[0], type(r[0]))
	print(r[0][0], type(r[0][0]))


	# s = 5
	# r = React(s)
	# print(r, type(r))

	#
	# s = [[3]]
	# #s = [3]
	# r = React(s)
	# #r = React.__new__(React, s) # THIS WORKS ._.
	# log(r, type(r))
	# log(r)
	# print(r[0], type(r[0]))
	# print(r[0][0], type(r[0][0]))
	# pass
