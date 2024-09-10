
from __future__ import annotations

import pprint
import typing as T
from collections import defaultdict

from wplib import inheritance, dictlib, log
from wplib.serial import serialise, deserialise, Serialisable, SerialAdaptor
from wplib.object import DeepVisitor, Adaptor
from wpdex.base import WpDex, DexPathable

class ReactiveDeserialiseOp(DeepVisitor.DeepVisitOp):
	@classmethod
	def visit(cls,
	          obj:T.Any,
	              visitor:DeepVisitor=None,
	              visitObjectData:VisitObjectData=None,
	              #visitPassParams:VisitPassParams=None,
	              ) ->T.Any:

		"""Transform to apply to each object during deserialisation.
		"""
		result = None
		visitPassParams:VisitPassParams = visitObjectData["visitPassParams"]

		serialParams : dict = dict(visitPassParams.visitKwargs["serialParams"])

		if obj is None:
			return None

		if not isinstance(obj, dict):
			result = obj

		elif obj.get(SerialAdaptor.FORMAT_KEY, None) is None:
			result = obj
		log("VISIT", obj, type(obj), result)

		if result is not None: # it's a primitive or a simple dict,
			# copy by direct initialisation
			# TODO: shuffle the __new__ of the generated class so we don't get a loop
			#  when base class is called
			serialisedType = type(obj)
			reactType = React.getProxyTypeForOrigType(serialisedType)
			log("reactType", reactType, reactType.__new__)
			reactObj = reactType(obj)

		else:

			# reload serialised type and get the adaptor for it
			serialisedType = SerialAdaptor.typeForData(obj[SerialAdaptor.FORMAT_KEY])
			# get the react type for this loaded type
			reactType = React.getProxyTypeForOrigType(serialisedType)

			# still use the normal deserialise adaptor to load in the data
			adaptorCls = SerialAdaptor.adaptorForType(serialisedType)

			if adaptorCls is None:
				raise Exception(f"No adaptor for class {type(obj)}")
			dictlib.defaultUpdate(serialParams, adaptorCls.defaultDecodeParams(serialisedType)
			                      )
			# decode the object
			log("decode", obj, adaptorCls)
			reactObj = adaptorCls.decode(
				obj,
				#serialType=serialisedType,
				serialType=reactType,
				decodeParams=serialParams)
		reactObj.__reactInit__(obj)
		log("deserialised obj", reactObj, type(reactObj))
		return reactObj


class ReactCopyOp(DeepVisitor.DeepVisitOp):
	@classmethod
	def visit(cls,
	          obj:T.Any,
	              visitor:DeepVisitor=None,
	              visitObjectData:VisitObjectData=None,
	              #visitPassParams:VisitPassParams=None,
	              ) ->T.Any:

		"""Transform to apply to each object during deserialisation.
		"""
		visitPassParams:VisitPassParams = visitObjectData["visitPassParams"]

		serialParams : dict = dict(visitPassParams.visitKwargs["serialParams"])

		if obj is None:
			return None

		return React._copy(obj)

class ReactLogic(Adaptor):
	"""silo off the active parts of react to avoid jamming in
	even more base classes, it's getting mental"""
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	forTypes = (object, )


class ReactMeta(type):

	def __call__(cls, *args, **kwargs):
		log("META call", cls, args, kwargs)
		try:
			result = type.__call__(cls, *args, **kwargs)
		except TypeError:
			cls.__new__ = cls.__mro__[1].__new__
			result = type.__call__(cls, *args, **kwargs)
		log("  META call result", result)
		return result

class React(metaclass=ReactMeta):
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
	"""
	_classProxyCache : dict[type, dict[type, type]] = defaultdict(dict) # { class : { class cache } }

	_proxyAttrs = {"_reactData" : {}}
	_generated = False
	_objIdProxyCache : dict[int, React] = {}


	def __reactInit__(self, obj, reactData:dict=None, **kwargs):
		"""called by __new__, after object has been deserialised
		STORE NO REFERENCE TO OBJ, the original object -
		this reactive object REPLACES it"""
		log("react INIT", obj, self, type(self))
		self._reactData = reactData

	@classmethod
	def _existingProxy(cls, obj):
		"""retrieve an existing proxy object for the given base object"""
		return cls._objIdProxyCache.get(id(obj))


	def _beforeProxyCall(self, methodName: str,
	                     methodArgs: tuple, methodKwargs: dict,
	                     targetInstance: object
	                     ) -> tuple[T.Callable, tuple, dict, object]:
		"""overrides / filters args and kwargs passed to method
		getting _proxyBase and _proxyResult should both be legal here"""
		newMethod = getattr(targetInstance, methodName)
		# log("before proxy call", methodName, newMethod, methodArgs, methodKwargs, targetInstance)
		return newMethod, methodArgs, methodKwargs, targetInstance

	def _afterProxyCall(self, methodName: str,
	                    method: T.Callable,
	                    methodArgs: tuple, methodKwargs: dict,
	                    targetInstance: object,
	                    callResult: object,
	                    ) -> object:
		"""overrides / filters result of proxy method"""
		# log("after proxy call", methodName, method, methodArgs, methodKwargs, targetInstance, callResult)
		return callResult

	def _onProxyCallException(self,
	                          methodName: str,
	                          method: T.Callable,
	                          methodArgs: tuple, methodKwargs: dict,
	                          targetInstance: object,
	                          exception: BaseException) -> (None, object):
		"""called when proxy call raises exception -
		to treat exception as normal, raise it from this function as well
		if no exception is raised, return value of this function is used
		as return value of method call
		"""
		raise exception

	@classmethod
	def _makeProxyMethod(cls, methodName, targetCls):
		"""called with each method of the target cls,
		by default looks up proxy on each call
		default implementation provides hooks to override before and after
		call, also allowing filtering args, kwargs, as well as what method
		is actually called, and on what instance

		shifting to return the method object itself, not just the name
		"""

		def _proxyMethod(self: Proxy, *args, **kw):
			# pre-call hook - must return parametres of method call
			newMethod, newArgs, newKw, targetInstance = self._beforeProxyCall(
				methodName, methodArgs=args, methodKwargs=kw,
				targetInstance=self._proxyTarget()
			)

			try:  # set up exception catch
				# actually call method on target instance, get raw result
				result = newMethod(*newArgs, **newKw)
			except Exception as e:  # on any exception, pass to handler function
				result = self._onProxyCallException(
					methodName, newMethod,
					methodArgs=newArgs, methodKwargs=newKw,
					targetInstance=targetInstance,
					exception=e
				)

			# filter result based on call params, return filtered output
			newResult = self._afterProxyCall(
				methodName, newMethod,
				methodArgs=newArgs, methodKwargs=newKw,
				targetInstance=targetInstance,
				callResult=result)
			return newResult

		return _proxyMethod

	@classmethod
	def _createClassProxy(cls, targetCls):
		"""creates a proxy class for the given class

		this wraps all undefined methods with the before- and after- functions,
		and also handles setting the proxy attributes

		"""
		log("create class proxy for", targetCls)
		# build namespace for generated class
		# combine declared proxy attributes
		allProxyAttrs = set(cls._proxyAttrs)
		for base in cls.__mro__:
			if getattr(base, "_proxyAttrs", None):
				allProxyAttrs.update(base._proxyAttrs)

		# work out parent, super and target classes
		proxyClasses = [x for x in cls.__mro__ if issubclass(x, React) and not getattr(x, "_generated", False)][::-1]
		_proxyParentCls = proxyClasses[-1]
		_proxySuperCls = None
		if len(proxyClasses) > 1:
			_proxySuperCls = proxyClasses[-2]

		# namespace dict
		namespace = {"_proxyAttrs": allProxyAttrs,
		             "_generated": True,
		             "_proxyTargetCls": targetCls,
		             "_proxyParentCls": _proxyParentCls,
		             "_proxySuperCls": _proxySuperCls,
		             }
		#toWrap = inheritance.classCallables(targetCls)

		# for methodName in toWrap:
		# 	# do not override methods if they appear in proxy class
		# 	# unless they are in the "wrapTheseMethodsAnyway" list
		# 	# print("wrap", methodName, methodName in cls._wrapTheseMethodsAnyway, methodName in dir(cls))
		# 	if hasattr(targetCls, methodName):
		# 		if ((methodName in cls._wrapTheseMethodsAnyway)
		# 				or (not methodName in dir(cls))):
		# 			# print("wrapping", methodName)
		# 			namespace[methodName] = cls._makeProxyMethod(methodName, targetCls)

		#clsType = ProxyMeta
		clsType = type

		# proxying things like bools and ints is useful for persistence,
		# but precludes the more complex proxy wrapping and typing -
		# inherit directly from Proxy in these cases

		bases = (cls, targetCls)
		# try:
		# 	testType = clsType("test", bases, {})
		# except TypeError:
		# 	bases = (cls, )

		# generate new type inheriting from all relevant bases
		newCls = clsType("{}({})".format(cls.__name__, targetCls.__name__),
		                 bases,
		                 namespace)
		# set the __new__ of this type to that of the target, so we don't get
		# infinite loops throught the React __new__
		newCls.__new__ = targetCls.__new__
		return newCls


	@classmethod
	def getProxyTypeForOrigType(cls, objCls):
		"""returns the proxy type for the given original type"""
		# if obj is proxy, look at its type
		cache = cls._classProxyCache
		try:
			genClass = cache[cls][objCls]
		except KeyError:
			genClass = cls._createClassProxy(objCls)
			cache[cls] = {objCls: genClass}
		return genClass

	def __new__(cls, obj, params=None, *args, **kwargs):
		"""
        creates a copy of the passed structure -
        we cheap out by using __new__ on the class to return an existing object if found,
        rather than __call__ on the metaclass, but that also means we don't have to worry
        about metaclass clashes in our synthetic types

		"""
		uniqueId = id(obj)
		# sharing makes no sense here, since we want the actual object
		if isinstance(obj, React):
			return obj

		# serialise given structure, then deserialise with a custom class
		result = deserialise(serialise(obj),
		                   deserialiseOp=ReactiveDeserialiseOp)
		log("_new_ result", result)
		# if isinstance(result, list):
		# 	log("entry", result[0], type(result[0]))

		return result

	# @classmethod
	# def _copy(cls, obj):
	
	# def __str__(self):
	# 	return ("WRX({})".format(super().__str__()))
	def __repr__(self):
		return ("WRX({})".format(super().__repr__()))


	@classmethod
	def _getProxy(cls, obj,):
		"""amusingly this is called in the inverse order of the base proxy class

		React.__new__
		-> DeserialiseOp
			-> React._getProxy() # multiple times throughout hierarchy

		"""
		cache = inheritance.mroMergedDict(cls)["_classProxyCache"]
		# log("proxy new", cls, obj, type(obj), vars=0)

		# check that we don't start generating classes from generated classes
		cls = next(filter(lambda x: not getattr(x, "_generated", False),
		                  cls.__mro__))




		# if obj is proxy, look at its type
		# if React in type(obj).__mro__:
		# 	objCls = obj.__class__
		# else:
		# 	objCls = type(obj)
		# try:
		# 	genClass = cache[cls][objCls]
		# except KeyError:
		# 	genClass = cls._createClassProxy(objCls)
		# 	cache[cls] = {objCls: genClass}

		try:
			ins = object.__new__(genClass)
		except TypeError:
			# for builtins need to call:
			#  int.__new__( ourNewClass, 3 )
			ins = objCls.__new__(genClass, obj)

		proxyObj = ins
		proxyObj.__init__(obj, proxyData, **kwargs)
		# proxyObj._proxyStrongRef = targetObj
		log("insert obj id", uniqueId, targetObj)
		cls._objIdProxyCache[uniqueId] = proxyObj

		# proxyObj._proxyStrongRef = targetObj

	@classmethod
	def _callNewOnGenType(cls, obj, objCls, genClass):
		try:
			ins = object.__new__(genClass)
		except TypeError:
			# for builtins need to call:
			#  int.__new__( ourNewClass, 3 )
			ins = objCls.__new__(genClass, obj)
		return ins

if __name__ == '__main__':

	# s = 5
	# r = React(s)
	# print(r, type(r))

	#
	s = [[3]]
	#s = [3]
	#r = React(s)
	r = React.__new__(React, s)
	log(r, type(r))
	log(r)
	print(r[0], type(r[0]))
	print(r[0][0], type(r[0][0]))
	pass
