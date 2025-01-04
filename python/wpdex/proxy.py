
from __future__ import annotations

import pprint
import weakref
import types
import typing as T
from collections import defaultdict
from pathlib import Path

from wpdex.wx import WX, Wreactive_ops
from wpdex.context import ReentrantContext

from wplib import log
from wplib.object import Adaptor, Proxy, ProxyData, ProxyMeta
from wplib.serial import serialise, deserialise
from wpdex.base import WpDex

from param import rx
import param

#import json as _json
# try:
# 	import orjson as _json
# except ImportError:
	#pass
import orjson

setattr(param.reactive, "reactive_ops", Wreactive_ops)


class WpDexProxyData(ProxyData):
	parent : weakref.ref
	wpDex : WpDex
	deltaStartState : T.Any
	deltaCallDepth : int

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
	INTERNALLY, wrapped object should be able to verify itself
	with "is" and exact checks - shouldn't ever touch a proxy directly

	proxy only exists to capture calls and sets -
	proxy[0] just returns a simple proxy of that value, if it's already wrapped by a proxy/dex

	proxy.x()[0] returns a dynamic expression of looking up that value whenever it's evaluated (somehow unify with param.rx)

	"""
	_classProxyCache : dict[type, dict[type, type]] = defaultdict(dict) # { class : { class cache } }

	_proxyAttrs = {"_reactData" : {}}
	_generated = False
	_objIdProxyCache : dict[int, WpDexProxy] = {}


	def __init__(self, obj:VT, proxyData: dict=None,
	             wpDex:WpDex=None,# parentDex:WpDex=None,
	             **kwargs)->VT:
		"""create a WpDex object for this proxy, add
		weak sets for children and parent"""
		#log("WP init", obj, type(obj), type(self), self._proxyParentCls, self._proxySuperCls)
		self._proxySuperCls.__init__(self, obj, proxyData, **kwargs)

		self._proxyData["externalCallDepth"] = 0
		self._proxyData["deltaCallDepth"] = 0
		self._proxyData["wxRefs"] : dict[WpDex.pathT, WX] = {}
		self._proxyData["deltaContext"] : ReentrantContext = kwargs.pop("deltaContext", None)
		self._proxyData["filePath"] : Path = None # file for live-linking
		#self._proxyData["fileSerialParams"] : dict = None # file for live-linking

		#self._proxyData["parent"] = self._proxyData.get("parent", None)

		# wpdex set up
		if wpDex is not None: # pre-built dex passed in
			self._proxyData["wpDex"] = wpDex
		else:
			dexCls = WpDex.adaptorForObject(obj)
			#log("dexCls", dexCls, obj)
			self._proxyData["wpDex"] = WpDex(obj)
			# link parent dex if given

		#self._proxyData["branches"] = {}

	def deltaContext(self)->ReentrantContext:
		"""return a context object to use directly -
		with proxy.deltaContext() :
			etc
			"""
		if self._proxyData["deltaContext"] is None:
			self._proxyData["deltaContext"] = ReentrantContext(
				proxy=self,
				onTopEnterFn=self.dex().prepForDeltas,
				onTopExitFn=self._gatherEmitDeltas,
			)
		return self._proxyData["deltaContext"]

	def dex(self)->WpDex:
		return self._proxyData["wpDex"]

	def __str__(self):
		return "WPX(" + super().__str__() + ")"


	def __repr__(self):
		return repr(self._proxyTarget())

	# def __format__(self, format_spec):
	# 	pass

	def _openDelta(self):
		"""open a delta for this object -
		if mutating functions are reentrant, only open one delta"""
		if self._proxyData["deltaCallDepth"] == 0:
			#log("OPEN ", self, "prepped", self.dex().isPreppedForDeltas, self.dex().branches)
			if not self.dex().isPreppedForDeltas:
				self.dex().prepForDeltas()
		self._proxyData["deltaCallDepth"] += 1

	def _gatherEmitDeltas(self, *args, **kwargs):
		deltaMap = self.dex().gatherDeltas()
		if deltaMap:
			event = {"type": "deltas",
			         "paths": deltaMap}
			self.dex().sendEvent(event)

	def _emitDelta(self):
		"""emit a delta for this object"""
		#log("emitDelta", self._proxyTarget(), self._proxyTarget().childObjects({}))
		#log( " live dex", self.dex(), self.dex().branches)
		self._proxyData["deltaCallDepth"] -= 1
		if self._proxyData["deltaCallDepth"] == 0:
			# event = {"type":"change",
			#          "paths" : [()]}
			# self.dex().sendEvent(event)

			deltaMap = self.dex().gatherDeltas(
				emit=True
			)

			# if deltaMap:
			# 	event = {"type":"deltas",
			# 	         "paths" : deltaMap}
			# 	self.dex().sendEvent(event)



	def _beforeProxyCall(self, methodName:str,
	                     methodArgs:tuple, methodKwargs:dict,
	                     targetInstance:object
	                     )->tuple[T.Callable, tuple, dict, object, dict]:
		"""open a delta if mutating method called"""
		#log(f"before proxy call {methodName}, {methodArgs, methodKwargs}", vars=0)
		fn, args, kwargs, targetInstance, beforeData = super()._beforeProxyCall(methodName, methodArgs, methodKwargs, targetInstance)
		self._proxyData["externalCallDepth"] += 1

		# don't pass proxies into the base objects
		filterArgs = []
		for i in args:
			filterArgs.append(i._proxyTarget() if isinstance(i, Proxy) else i)
		filterKwargs = {}
		for k, v in kwargs.items():
			filterKwargs[k] = v._proxyTarget() if isinstance(v, Proxy) else v

		# check if method will mutate data - need to open a delta
		# unless context is already open
		if self._proxyData["deltaContext"] is not None:
			if self.deltaContext().depth != 0:
				return fn, tuple(filterArgs), filterKwargs, targetInstance, beforeData

		if methodName in self.dex().mutatingMethodNames:
			self._openDelta()

		return fn, tuple(filterArgs), filterKwargs, targetInstance, beforeData

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
		self._proxyData["externalCallDepth"] -= 1

		callResult = super()._afterProxyCall(
			methodName, method, methodArgs, methodKwargs, targetInstance, callResult,
			beforeData, exception
		)

		#log("after call", methodName, methodArgs, callResult, self._proxyData["externalCallDepth"], methodName in self.dex().mutatingMethodNames)
		#log("dex", self.dex())

		toReturn = callResult

		if methodName in self.dex().mutatingMethodNames:
			# ensure every bit of the structure is still wrapped in a prox
			self.dex().updateChildren(recursive=1)

		"""
		delta gathering on wpdex side works super well - all we need from the proxy is:
		- way to detect changes
		- way to associate objects with a WpDex (with one spot in hierarchy)
		- wrapper to set up rx expressions

		"""
		# NB: for now as a special case, we don't wrap returned types or functions
		# it gets very messy, and there's not a use case yet
		if isinstance(toReturn, (type, types.FunctionType)):
			return toReturn
		if self._proxyData["externalCallDepth"] != 0:
			return toReturn
		foundDex = WpDex.dexForObj(toReturn)
		#log("found dex", foundDex, toReturn,  self._proxyData["externalCallDepth"])
		if foundDex:
			if self._proxyData["externalCallDepth"] == 0:
				toReturn = WpDexProxy(toReturn, wpDex=foundDex)

		# check if a higher effect has a delta context open
		# if so, skip
		if self._proxyData["deltaContext"] is not None:
			if self.deltaContext().depth != 0:
				return toReturn

		# if mutating method called, finally gather and emit delta
		if self._proxyData["externalCallDepth"] == 0:
			if methodName in self.dex().mutatingMethodNames:
				# self._emitDelta()
				self._proxyData["deltaCallDepth"] -= 1
				if self._proxyData["deltaCallDepth"] == 0:

					deltaMap = self.dex().gatherDeltas()

					if deltaMap:
						#todo: an easy way in dex to just compare all child dex objects for combined deltas
						log("deltaMap")
						pprint.pprint(deltaMap)
						event = {"type":"deltas",
						         "paths" : deltaMap}
						self.dex().sendEvent(event)

		return toReturn

	def _beforeProxySetAttr(self, attrName:str, attrVal:T.Any,
	                     targetInstance:object
	                     ) ->tuple[str, T.Any, object, dict]:
		"""in general we assume setting an attribute will usually 
		mutate an object"""
		self._proxyData["externalCallDepth"] += 1
		self._openDelta()
		if isinstance(attrVal, Proxy):
			attrVal = attrVal._proxyTarget()
		return super()._beforeProxySetAttr(attrName, attrVal, targetInstance)

	def _afterProxySetAttr(self, attrName:str, attrVal:T.Any,
	                     targetInstance:object, beforeData:dict, exception=None
	                     ) ->None:
		#self.updateProxy()
		self.dex().updateChildren(recursive=1)
		self._proxyData["externalCallDepth"] -= 1
		self._emitDelta()
		return super()._afterProxySetAttr(attrName, attrVal, targetInstance, beforeData, exception)

	def _beforeProxyGetAttr(self, attrName:str,
	                     targetInstance:object
	                     ) ->tuple[str, object, dict]:
		self._proxyData["externalCallDepth"] += 1
		return super()._beforeProxyGetAttr(attrName, targetInstance)

	def _afterProxyGetAttr(self, attrName:str, attrVal:T.Any,
	                     targetInstance:object, beforeData:dict, exception=None
	                     ) ->T.Any:
		"""if it's an internal call, flatten result if it's a proxy"""
		result = super()._afterProxyGetAttr(attrName, attrVal, targetInstance, beforeData, exception)
		self._proxyData["externalCallDepth"] -= 1
		if self._proxyData["deltaCallDepth"] > 0:
			if isinstance(result, WpDexProxy):
				result = result._proxyTarget()
		else:
			# only wrap part of the data structure, as mapped out by WpDex,
			# not just any random attr / property that could be procedural
			foundDex = WpDex.dexForObj(result)
			if foundDex:
				result = WpDexProxy(result, wpDex=foundDex)
		return result


	@classmethod
	def _setProxyTarget(cls, proxy:WpDexProxy, target):
		super()._setProxyTarget(proxy, target)
		if not "wpDex" in proxy._proxyData:
			return
		#proxy.dex().obj = target
		proxy.dex().setObj(target, )
		#proxy.dex().updateChildren()

		#proxy._linkDexProxyChildren()

	#region reactive referencing
	def ref(self, *path:WpDex.pathT)-> WX:
		"""unsure if this should return a wx on the TARGET object,
		or on the FUNCTION to GET the target object
		hmmmmmmm

		test returning a lifted call, on the method to get the target.
		every time we recompute the rx expression result, we have to start
		by retrieving the live value of the structure at this path.
		wew lad
		"""
		#TODO: factor this out properly, MAYBE as a classmethod on WX?


		#TODO: couldn't get rx.when() to work for gating eval,
		# it would always return an rx() instance, not WX(),
		# so for this case I use a dummy argument to the function itself
		path = WpDex.toPath(path)
		#log("get ref", self, path)
		if self._proxyData["wxRefs"].get(path) is None:

			dirtyKwarg = rx(1)

			def _resolveRef(**kwargs):
				rootDex = self.dex()
				foundDex : WpDex = rootDex.access(rootDex, path, values=False, one=True,
				                                  )

				return foundDex.getValueProxy()

				pass

			ref = WX(_resolveRef, _dexPath=path, _dex=self.dex())(dirtyKwarg=dirtyKwarg)

			assert isinstance(ref, WX)
			assert isinstance(ref.rx._reactive, WX)

			self._proxyData["wxRefs"][path] = ref
			# flag that it should dirty whenever this proxy's
			# value changes (on delta)
			def _setDirtyRxValue(*_, **__):
				"""need to make a temp closure because we can't
				easily set values as a function call"""
				#log("set dirty value")
				dirtyKwarg.rx.value += 1

			self.dex().getEventSignal("main").connect(_setDirtyRxValue)

			# allow writing back by "WRITE" method on WX
			# TODO: maybe move more of this into WX, pass in reference to wpdex root?

			def _onRefWrite(path, value):
				#log("start dex", self.dex(), self.dex().parent, self.dex().branchMap())
				#log(self.dex().branchMap()["@N"].parent)
				dex = self.dex()
				#log("path", path, self.dex().toPath(path))
				targetDex : WpDex = dex.access(dex, self.dex().toPath(path), values=False)
				#log("targetDex", targetDex, targetDex.parent)
				#log(targetDex.branchMap())
				assert isinstance(targetDex, WpDex)
				if targetDex.parent is not None:
					assert isinstance(targetDex.parent, WpDex)
				targetDex.write(value)
				#log("after write", self.dex(), value, type(value), self.dex().branchMap())
			ref._kwargs["_writeSignal"].connect(_onRefWrite)

		return self._proxyData["wxRefs"][path]


	#endregion

	def linkToFile(self, path:Path, serialParams=None):
		self._proxyData["filePath"] = path
		self._proxyData["fileSerialParams"] = serialParams
		self.dex().getEventSignal("main").connect(
			lambda *a, **kw : self.writeToFile(serialParams=serialParams))
	def writeToFile(self, path:Path=None, serialParams=None):
		targetPath = path or self._proxyData["filePath"]
		serialParams = serialParams or self._proxyData.get("serialParams")
		assert targetPath, f"Must give or previously pair a valid path to serialise, not {targetPath}"
		with open(targetPath, "wb") as f:
			f.write(orjson.dumps(self.dex().serialiseData(serialParams)))

	def readFromFile(self, path:Path=None, serialParams=None):
		targetPath = path or self._proxyData["filePath"]
		serialParams = serialParams or self._proxyData.get("serialParams")
		assert targetPath, f"Must give or previously pair a valid path to deserialise, not {targetPath}"
		assert targetPath.exists(), f"no path found at {targetPath} to deserialise {self}"
		with open(targetPath, "rb") as f:
			data = orjson.loads(f.read())
		obj = deserialise(data, serialParams=serialParams)
		self._setProxyTarget(self, obj)

	@classmethod
	def fileLinkedObject(cls, defaultObj, path:Path,
	                     readOnStart=True,
	                     writeOnStart=False,
	                     serialParams=None):
		proxy = WpDexProxy(defaultObj)
		proxy.linkToFile(path, serialParams=serialParams)
		if path.exists() and readOnStart: #
			proxy.readFromFile()
		if writeOnStart:
			proxy.writeToFile()
		return proxy




if __name__ == '__main__':
	s = {"a" : 2,
	     "b" : [4, 5, 6],
	     "c" : 3
	     }
	p = WpDexProxy(s)
	log("p", p)
	log(p.dex(), p.dex().branchMap())
	log(p.dex().access(p.dex(), "b", values=True))
	log(p.dex().allBranches())
	r = p.ref("b")
	print(r)
	p["b"][2] = "hello"
	p["b"] = "hello"

	r.rx.watch(lambda x : print("changed", x))

	p["b"] = "WOW"
	p["b"] = [1, 2, 3]
	p["b"][1] = "WOOOOOOW"



