
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
		#log("WPMETA call", cls, args, type(args[0]), kwargs)
		parent = kwargs.pop("parent", None)
		isRoot = kwargs.pop("isRoot", None)

		# called at top level, wrap hierarchy
		# if isRoot is None and parent is None:
			#return cls.wrap(args[0], **kwargs)
		return super().__call__(*args, **kwargs)


VT = T.TypeVar("VT")
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

	now go back to interface layer idea, rather than embedding proxies into hierarchy -
	INTERNALLY, wrappe object should be able to verify itself
	with "is" and exact checks - shouldn't ever touch a proxy directly

	proxy only exists to capture calls and sets -
	proxy[0] just returns a simple proxy of that value, if it's already wrapped by a proxy/dex

	proxy.x()[0] returns a dynamic expression of looking up that value whenever it's evaluated (somehow unify with param.rx)

	"""
	_classProxyCache : dict[type, dict[type, type]] = defaultdict(dict) # { class : { class cache } }

	_proxyAttrs = {"_reactData" : {}}
	_generated = False
	_objIdProxyCache : dict[int, WpDexProxy] = {}

	def __init__(self, obj:VT, proxyData: WpDexProxyData=None,
	             wpDex:WpDex=None,# parentDex:WpDex=None,
	             **kwargs)->VT:
		"""create a WpDex object for this proxy, add
		weak sets for children and parent"""
		#log("WP init", obj, type(obj), type(self), self._proxyParentCls, self._proxySuperCls)
		self._proxySuperCls.__init__(self, obj, proxyData, **kwargs)


		self._proxyData["deltaStartState"] = None
		self._proxyData["deltaCallDepth"] = 0
		self._proxyData["externalCallDepth"] = 0

		self._proxyData["parent"] = self._proxyData.get("parent", None)

		# wpdex set up
		if wpDex is not None: # pre-built dex passed in
			self._proxyData["wpDex"] = wpDex
		else:
			self._proxyData["wpDex"] = WpDex(obj)
			# link parent dex if given

		self._proxyData["branches"] = {}

		#self.dex().updateChildren()
		self.updateProxy()


	def dex(self)->WpDex:
		return self._proxyData["wpDex"]

	def __str__(self):
		return "WPX(" + super().__str__() + ")"

	# def __repr__(self):
	# 	return "WPX(" + repr(self._proxyTarget()) +


	def _openDelta(self):
		"""open a delta for this object -
		if mutating functions are reentrant, only open one delta"""
		if self._proxyData["deltaCallDepth"] == 0:
			#self._proxyData["deltaStartState"] = copy.deepcopy(self._proxyData["target"])
			# self._proxyData["deltaStartState"] = serialise(
			# 	self._proxyCleanResult(),
			# 	serialParams={"TreeSerialiseUid" : False}
			# )
			log("OPEN ", self, "prepped", self.dex().isPreppedForDeltas, self.dex().branches)
			if not self.dex().isPreppedForDeltas:
				self.dex().prepForDeltas()
		self._proxyData["deltaCallDepth"] += 1

	def _emitDelta(self):
		"""emit a delta for this object"""
		log("emitDelta", self._proxyTarget(), self._proxyTarget().childObjects({}))
		log( " live dex", self.dex(), self.dex().branches)
		self._proxyData["deltaCallDepth"] -= 1
		if self._proxyData["deltaCallDepth"] == 0:
			# self._proxyData["deltaEndState"] = serialise(
			# 	self._proxyCleanResult(),
			# 	serialParams={"TreeSerialiseUid": False}
			# )

			deltaMap = self.dex().gatherDeltas(
				# self._proxyData["deltaStartState"],
				# self._proxyData["deltaEndState"],
			)
			log("WPX", self, "result deltaMap", deltaMap)
			#if deltaMap:
			# delta = DeltaAtom(self._proxyData["deltaStartState"], self._proxyData["target"])
			self._proxyData["deltaStartState"] = None
			self._proxyData["deltaEndState"] = None

			if deltaMap:
				event = {"type":"deltas",
				         "paths" : deltaMap}
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
	                     )->tuple[T.Callable, tuple, dict, object, dict]:
		"""open a delta if mutating method called"""
		#log(f"before proxy call {methodName}, {methodArgs, methodKwargs}", vars=0)
		fn, args, kwargs, targetInstance, beforeData = super()._beforeProxyCall(methodName, methodArgs, methodKwargs, targetInstance)
		self._proxyData["externalCallDepth"] += 1

		# if not self.dex().childIdDexMap:
		# 	self.dex().updateChildren()
		#
		# check if method will mutate data - need to open a delta
		if methodName in self.dex().mutatingMethodNames:
			self._openDelta()

		return fn, args, kwargs, targetInstance, beforeData

	def _afterProxyCall(self, methodName:str,
	                    method:T.Callable,
	                     methodArgs:tuple, methodKwargs:dict,
	                    targetInstance: object,
	                    callResult:object,
	                    beforeData:dict,
	                    exception:BaseException=None
	                    )->object:
		"""return result wrapped in a wpDex proxy, if it appears in
		main dex children

		gather deltas, THEN refresh children, THEN emit deltas

		TODO: should we check validation here, or should there be a separate
		 signal / event before the call to check arguments
		"""
		callResult = super()._afterProxyCall(
			methodName, method, methodArgs, methodKwargs, targetInstance, callResult,
			beforeData, exception
		)
		self._proxyData["externalCallDepth"] -= 1


		toReturn = callResult

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

			# ensure every bit of the structure is still wrapped in a proxy
			self.updateProxy()

			#self.dex().updateChildren()


		# if mutating method called, finally emit delta
		if methodName in self.dex().mutatingMethodNames:
			self._emitDelta()

		# consider instance methods that "return self"
		# check if a proxy already exists for that object and return it
		checkExisting = self._existingProxy(toReturn)
		if checkExisting is not None:
			toReturn = checkExisting
		return toReturn

	def _beforeProxySetAttr(self, attrName:str, attrVal:T.Any,
	                     targetInstance:object
	                     ) ->tuple[str, T.Any, object, dict]:
		"""in general we assume setting an attribute will usually 
		mutate an object"""
		self._proxyData["externalCallDepth"] += 1
		self._openDelta()
		return super()._beforeProxySetAttr(attrName, attrVal, targetInstance)

	def _afterProxySetAttr(self, attrName:str, attrVal:T.Any,
	                     targetInstance:object, beforeData:dict, exception=None
	                     ) ->None:
		self.updateProxy()
		self._proxyData["externalCallDepth"] -= 1
		self._emitDelta()
		return super()._afterProxySetAttr(attrName, attrVal, targetInstance, beforeData, exception)
	
	# def _beforeProxyGetAttr(self, attrName:str,
	#                      targetInstance:object
	#                      ) ->tuple[str, object, dict]:
	# 	self._proxyData["externalCallDepth"] += 1
	# 	return super()._beforeProxyGetAttr(attrName, targetInstance)
	#
	# def _afterProxyGetAttr(self, attrName:str, attrVal:T.Any,
	#                      targetInstance:object, beforeData:dict, exception=None
	#                      ) ->T.Any:
	# 	"""if it's an internal call, flatten result if it's a proxy"""
	# 	result = super()._afterProxyGetAttr(attrName, attrVal, targetInstance, beforeData, exception)
	# 	self._proxyData["externalCallDepth"] -= 1
	# 	if self._proxyData["deltaCallDepth"] > 0:
	# 		if isinstance(result, WpDexProxy):
	# 			result = result._proxyTarget()
	# 	return result


	def updateProxy(self):
		""" we can't keep regenerating new proxy objects or no references will
		ever be stable
		we can't just update objects directly because immutable types exist

		compare CHILD objects, find any that are NOT PROXIES
		create proxies of them,
		update the TARGET OF THIS PROXY to the newObj(), formed by THOSE PROXIES.

		before update, save each proxy and its value against its path on the root

		child objects have to be the same between proxy and wpdex, insane otherwise

		"""
		log("update proxy", self)
		adaptor = VisitAdaptor.adaptorForObject(self._proxyData["target"])
		childObjects = list(adaptor.childObjects(self._proxyData["target"], {}))
		#childObjects = { k : v.obj for k, v in self.dex().keyDexMap.items()}
		# returns list of 3-tuples (index, value, childType)
		needsUpdate = False
		#for i, (k, t) in enumerate(childObjects.items()):
		for i, t in enumerate(childObjects):
			#log(" tie", t)
			if isinstance(t[1], WpDexProxy):
				# #raise RuntimeError
				# t[1]._proxyData["parent"] = self
				# #t[1].dex().parent = self.dex()
				# self.dex().addBranch(t[1].dex(), key=(t[0], ))
				# t[1].updateProxy()
				pass
			else:
				needsUpdate = True

				# ADD IN PATH LOOKUP HERE to check if proxy is already known
				log("wrap child", t[1])
				proxy = WpDexProxy(t[1], isRoot=False,
				                   proxyData={"parent" : self})
				if proxy is not None: # if you wrap a proxy of None
					#log(" add child proxy", self, proxy.dex(), t)
					self.dex().addBranch(proxy.dex(), key=(t[0], ))
					#proxy.dex().updateChildren()

				childObjects[i] = (t[0], proxy, t[2])
		if needsUpdate:
			"""this WORKS, and doesn't break external references, since the refs are
			to THIS proxy object, not the internal one being changed here.
			all the child refs survive as well"""
			#log("   updating", self, childObjects)
			#log([type(i[1]) for i in childObjects])
			newObj = adaptor.newObj(
				self._proxyData["target"], childObjects, params={"PreserveUid" : True})
			#log("final childObjects", adaptor.childObjects(newObj, {}))
			#log([type(i[1]) for i in adaptor.childObjects(newObj, {})])
			self._setProxyTarget(self, newObj)

		for i in adaptor.childObjects(self._proxyData["target"], {}):
			if i[1] is None:
				continue


		#TODO: WHERE TO RECURSE DEX, UPDATE CHILDREN????


		#self.dex().updateChildren()

			# if isinstance(i[1], WpDexProxy):
			#
			# 	# maybe we don't need to proxy primitive types?
			#
			# 	i[1].updateProxy()

	@classmethod
	def _setProxyTarget(cls, proxy:WpDexProxy, target):
		super()._setProxyTarget(proxy, target)
		if not "wpDex" in proxy._proxyData:
			return
		proxy.dex().obj = target
		proxy._linkDexProxyChildren()

	def _linkDexProxyChildren(self):
		"""run at end to regenerate links between structures -
		assumes that all child objects are already wrapped in proxies"""
		log("link dex proxies", self, self.dex())
		self.dex().updateChildren()
		adaptor = VisitAdaptor.adaptorForObject(self._proxyTarget())
		childObjects = list(adaptor.childObjects(self._proxyTarget(), {}))
		log(childObjects)
		for k, v, data in childObjects:
			log("child", k, (k, ) in self.dex().keyDexMap, v, type(v))
			if (k, ) in self.dex().keyDexMap:
				if v is None: continue
				#if not isinstance(v, WpDexProxy): continue
				#assert isinstance(v, WpDexProxy)
				v._proxyData["wpDex"] = self.dex().keyDexMap[(k, )]
				v._linkDexProxyChildren()


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
