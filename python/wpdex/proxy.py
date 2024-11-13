
from __future__ import annotations

import pprint, copy, weakref
import types
import typing as T
from collections import defaultdict
from dataclasses import dataclass

from wplib import inheritance, dictlib, log, sequence
from wplib.object import Signal
from wplib.serial import serialise, deserialise, Serialisable, SerialAdaptor
from wplib.object import DeepVisitor, Adaptor, Proxy, ProxyData, ProxyMeta, VisitObjectData, DeepVisitOp, Visitable, VisitAdaptor
from wplib.constant import SEQ_TYPES, MAP_TYPES, STR_TYPES, LITERAL_TYPES, IMMUTABLE_TYPES
from wplib.sentinel import Sentinel
from wpdex.base import WpDex

from param import rx
from param.reactive import reactive_ops, Parameter, resolve_value, resolve_ref
import param

class Wreactive_ops(reactive_ops):
	"""test overriding reactive_ops,
	more specific support for writing back to refs
	using
	.rx.value =
	syntax, since otherwise you have to do some type checking to
	tell when to use BIND(), etc
	"""
	@property
	def value(self):
		"""
		Returns the current state of the reactive expression by
		evaluating the pipeline.
		"""
		if isinstance(self._reactive, rx):
			return self._reactive._resolve()
		elif isinstance(self._reactive, Parameter):
			return getattr(self._reactive.owner, self._reactive.name)
		else:
			return self._reactive()

	@value.setter
	def value(self, new):
		"""
		Allows overriding the original input to the pipeline.
		"""
		# log("set value", new,
		#     self._reactive, self._reactive._wrapper)
		#root = self._reactive._compute_root()
		#log("root", root)

		print("")
		log("set value")
		wx = WX.getWXRoot(self._reactive)
		if wx is not None:
			#assert wx
			#if "_dexPath" in wx._kwargs:
			#if "_dexPath" in root._kwargs:
			#root.WRITE(resolve_value(new))
			log("EMITTING")
			wx.WRITE(resolve_value(new))
			return
		reactive_ops.value.fset(self, new) 

		# try:
		#
		# except AttributeError as e:
		# 	#return
		#
		#
		# 	hasPath = "_dexPath" in root._kwargs
		# 	log("root", root, hasPath)
		# 	if hasPath: return
		# 	raise e



	# @value.setter
	# def value(self, new):
	# 	"""
	# 	Allows overriding the original input to the pipeline.
	# 	"""
	#
	# 	log("set value", new,
	# 	    type(self._reactive), self._reactive._wrapper)
	# 	if "_dexPath" in self._reactive._kwargs:
	# 		self._reactive.WRITE(resolve_value(new))
	# 		return
	# 	rootHasPath = "_dexPath" in self._reactive._compute_root()._kwargs
	# 	if isinstance(self._reactive, Parameter):
	# 		raise AttributeError(
	# 			"`Parameter.rx.value = value` is not supported. Cannot override "
	# 			"parameter value."
	# 		)
	# 	elif not isinstance(self._reactive, rx):
	# 		raise AttributeError(
	# 			"`bind(...).rx.value = value` is not supported. Cannot override "
	# 			"the output of a function."
	# 		)
	# 	# elif "_dexPath" in self._reactive._kwargs:
	# 	# 	self._reactive.WRITE(resolve_value(new))
	# 	# 	return
	# 	elif self._reactive._root is not self._reactive:
	# 		if rootHasPath: return
	# 		raise AttributeError(
	# 			"The value of a derived expression cannot be set. Ensure you "
	# 			"set the value on the root node wrapping a concrete value, e.g.:"
	# 			"\n\n    a = rx(1)\n    b = a + 1\n    a.rx.value = 2\n\n "
	# 			"is valid but you may not set `b.rx.value = 2`."
	# 		)
	# 	if self._reactive._wrapper is None:
	# 		if rootHasPath: return
	# 		raise AttributeError(
	# 			"Setting the value of a reactive expression is only "
	# 			"supported if it wraps a concrete value. A reactive "
	# 			"expression wrapping a Parameter or another dynamic "
	# 			"reference cannot be updated."
	# 		)
	# 	self._reactive._wrapper.object = resolve_value(new)


setattr(param.reactive, "reactive_ops", Wreactive_ops)

class WX(rx):
	"""By default, printing an rx object freaks out because it returns
	an implicit __str__() call, not an actual string that can be printed

	putting in caps so that it's obvious when we do stuff with it
	"""
	def __repr__(self):
		try:
			return f"WX({repr(self.rx.value)})"
		except Exception as e:
			return f"WX(ERROR GETTING REPR)"
	def __str__(self):
		return f"WX({repr(self.rx.value)})"
	def __init__(self, *args,# path:WpDex.pathT=(),
	             **kwargs):
		if kwargs.get("_writeSignal", None) is None:
			kwargs["_writeSignal"] = Signal("WX-write")
		super().__init__(*args,# _dexPath=path,
		                 **kwargs
		                 )
		# pack path in _kwargs to ensure it gets copied on _clone()
		# also signal, they're expensive to build

	# TODO###: patch this in properly
	#  currently we have a filthy untracked patch in rx, to paste this block on the end of
	#   rx.__getattribute__
	#   instead of erroring straight out - waits to error until a reference actually computes

	# def __getattribute__(self, name):
	# 	try:
	# 		return super().__getattribute__(name)
	# 	except AttributeError:
	# 		#selfName = str(self).rx.value
	# 		# selfName = str(self)
	# 		# log("attribute error getting", name, "from", selfName)
	# 		new = self._resolve_accessor()
	# 		new._method = name
	# 		return new

	@classmethod
	def getWXRoot(cls, rxInstance:rx):
		while not isinstance(rxInstance, WX):
			try:
				log("discount rx", rxInstance)
			except:
				log("discount rx", type(rxInstance))
			rxInstance = rxInstance._prev
			if rxInstance is None: return None
		return rxInstance if isinstance(rxInstance, WX) else None

	def WRITE(self, val):
		"""emit (path, value)
		"""
		self._kwargs["_writeSignal"].emit(self._kwargs["_dexPath"], val)

	# def _resolveRef():
	# 	rootDex = self.dex()
	# 	foundDex: WpDex = rootDex.access(rootDex, path, values=False, one=True)
	# 	return foundDex.getValueProxy()

	def RESOLVE(self, dex=False, proxy=False, value=False):
		assert dex or proxy or value
		rootDex = self._kwargs["_dex"]
		path = self._kwargs["_dexPath"]
		if dex:
			return rootDex.access(rootDex, path, values=False, one=True)
		if proxy:
			return rootDex.access(rootDex, path, values=False, one=True).getValueProxy()
		if value:
			return self.rx.value



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

		self._proxyData["externalCallDepth"] = 0
		self._proxyData["deltaCallDepth"] = 0
		self._proxyData["wxRefs"] : dict[WpDex.pathT, WX] = {}

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
			#log("WPX", self, "result deltaMap", deltaMap)

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

		callResult = super()._afterProxyCall(
			methodName, method, methodArgs, methodKwargs, targetInstance, callResult,
			beforeData, exception
		)

		self._proxyData["externalCallDepth"] -= 1
		#log("after call", methodName, methodArgs, callResult, self._proxyData["externalCallDepth"], methodName in self.dex().mutatingMethodNames)
		#log("dex", self.dex())

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
			#self.updateProxy()
			#log(" send dex update children")
			self.dex().updateChildren(recursive=1)


		# if mutating method called, finally emit delta
		if methodName in self.dex().mutatingMethodNames:
			self._emitDelta()

		# consider instance methods that "return self"
		# check if a proxy already exists for that object and return it
		# checkExisting = self._existingProxy(toReturn)
		# if checkExisting is not None:
		# 	toReturn = checkExisting

		# get existing dex for this proxy if found
		"""
		delta gathering on wpdex side works super well - all we need from the proxy is:
		- way to detect changes
		- way to associate objects with a WpDex (with one spot in hierarchy)
		- wrapper to set up rx expressions
		
		if it weren't for immutable numbers and interning this would LITERALLY
		just work - but of course, there is no way to tell one True apart from another. Anywhere we can use id(), we cruise.
		SO we do some more digging
		
		
		ok for now we just can't do it.
		if you want to set up an expression from an integer, you start by referencing that
		point in hierarchy with a path, and go from there.
		for any immutable object, we just can't do it.
		
		BECAUSE the alternative (as we have right now) is inserting proxy objects directly
		between the existing objects, regenerating new objects all the time, 
		all that crazy stuff. 
		The example of Tree.commonParent() checks if "self.root is otherBranch.root" -
		totally sane and readable way of coding, and with proxies it's 50/50.
		for immutable values, the way to associate them consistently would be to wrap literally every attribute of every object throughout the structure, even the internal ones.
		(I think this is actually what rx does, but shockingly a commercial software company have done a better job)
		 
		within a proxy environment, who knows 
		"""
		foundDex = WpDex.dexForObj(toReturn)
		#log("found dex", foundDex, toReturn,  self._proxyData["externalCallDepth"])
		if foundDex:
			if self._proxyData["externalCallDepth"] == 0:

				#NB: for now as a special case, we don't wrap returned types or functions
				# it gets very messy, and there's not a use case yet
				if not isinstance(toReturn, (type, types.FunctionType)):
					toReturn = WpDexProxy(toReturn, wpDex=foundDex)
				#log("returning proxy", toReturn)

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
			foundDex = WpDex.dexForObj(result)
			if foundDex:
				result = WpDexProxy(result, wpDex=foundDex)
		return result


	@classmethod
	def _setProxyTarget(cls, proxy:WpDexProxy, target):
		super()._setProxyTarget(proxy, target)
		if not "wpDex" in proxy._proxyData:
			return
		proxy.dex().obj = target
		proxy.dex().updateChildren()
		#proxy._linkDexProxyChildren()

	#region reactive referencing
	def ref(self, *path:WpDex.pathT)->WX:
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
			#ref = WX(self._proxyTarget, path=path)()
			#if 1: self.dex().access()
			#dirtyRef = rx(path)
			dirtyKwarg = rx(1)
			# ref = WX(self.dex().access, _dexPath=path)(
			# 	obj=self.dex(),
			# 	#path=dirtyRef,
			# 	path=path,
			# 	values=True,
			# 	dirtyKwarg=dirtyKwarg
			# )
			# ref = WX(self.dex().access, _dexPath=path)(
			# 	obj=self.dex(),
			# 	# path=dirtyRef,
			# 	path=path,
			# 	values=True,
			# 	dirtyKwarg=dirtyKwarg
			# #).rx.pipe(WpDexProxy)
			# ).rx.pipe(WpDexProxy)

			def _resolveRef():
				rootDex = self.dex()
				foundDex : WpDex = rootDex.access(rootDex, path, values=False, one=True)

				return foundDex.getValueProxy()

				pass

			ref = WX(_resolveRef, _dexPath=path, _dex=self.dex())()

			assert isinstance(ref, WX)
			assert isinstance(ref.rx._reactive, WX)

			#log("ref is", ref)
			#log("ref call", type(ref), ref._obj, ref._operation)

			#ref.rx.when(dirtyKwarg)

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
				log("after write", self.dex(), self.dex().branchMap())
			ref._kwargs["_writeSignal"].connect(_onRefWrite)

		return self._proxyData["wxRefs"][path]


	#endregion



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



