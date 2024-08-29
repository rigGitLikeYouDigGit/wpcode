from __future__ import annotations
from collections import defaultdict
import typing as T

from wplib import log, inheritance
"""

I have split up the proxy functionality like so:
 - main Proxy class handles direct attribute lookup on the
target object and type
eg
objA = ["test"]
objAProxy = Proxy.getProxy(objA)
# look up the __len__ function on the proxy
len(objAProxy)
# base Proxy class does not override __len__
# call is delegated to objAProxy._proxyResult object (here the original list object)
>>> 1

# if a custom Proxy class overrides a function,
# a call to that function is intercepted
# we can also set and access _proxyData, a dict of user information
class ListProxy(Proxy):
	def __len__(self):
		return len(self._proxyResult()) + 1 + self._proxyData["padding"]

objAListProxy = ListProxy.getProxy(
	objA, 
	proxyData={"padding" : 2}
	)
len(objAListProxy)
>>> 4

LinkedProxy allows a surface Proxy object to be "retargeted" at a different base object,
totally invisibly to the client code
- ProxyLink class handles logic to find this new target
target between the base and result
also avoids polluting proxy namespace too much


probably the most complex thing I've ever done, but with hindsight I think it's actually pretty cool


"""


def classCallables(cls, dunders=True):
	"""return all attributes of a class that can be called"""
	methodNames = []
	for attrName in dir(cls):
		attrValue = getattr(cls, attrName)
		if callable(attrValue):
			if any((dunders, not attrName.startswith("__"))):
				methodNames.append(attrName)
	return methodNames





class ProxyMeta(type):
	"""used only for subclasscheck misdirection"""

	if not T.TYPE_CHECKING:
		def __call__(cls:type[Proxy], obj, **kwargs):
			#print("meta call", cls, obj, kwargs)
			return cls.getProxy(obj, **kwargs)

class ProxyData(T.TypedDict):
	"""user data for a proxy object"""
	target : object
	# other data

class Proxy(#ABC,
	#object,
    metaclass=ProxyMeta
            ):
	""" Transparent proxy for most objects
	code recipe 496741
	further modifications by ya boi

	__call__, then __new__, then __init__

	some of the mechanisms could move to ProxyMeta metaclass,
	but it seems to work fine as it is

	proxy object stores a persistent reference to its target
	object, and provides an overridable function to return it


	"""
	_classProxyCache = defaultdict(dict) # { class : { class cache } }

	_generated = False
	_objProxyCache = {} #NB: weakdict was giving issues, this might chug memory
	#_proxyLinkCls = None # optionally enforce a link class for this class of proxy

	# define explicit list of attributes on proxy object, like __slots__
	_proxyAttrs = (
		"_proxyData",
		#"_proxyLink",
       # "_proxyResult", "_proxyBase", "_proxyStrongRef"
	               )

	# type assistance with proxy data dict
	_proxyData : ProxyData

	_proxyParentCls : type[Proxy]
	_proxySuperCls : (type[Proxy], None)
	_proxyTargetCls : T.Type[object] = None # type this proxy is wrapping

	# def __init_subclass__(cls, **kwargs):
	# 	"""get the parent and super proxy classes
	#
	# 	TODO: we could put this into createClassProxy() -
	# 	 only advantage of init_subclass is, it sets the attributes on
	# 	 the actual proxy class, not just the generated ones
	#
	# 	 this may never be relevant
	# 	"""
	# 	super().__init_subclass__(**kwargs)
	# 	mro = cls.__mro__
	# 	print("init subclass", cls, mro)
	# 	proxyClasses = [x for x in cls.__mro__ if issubclass(x, Proxy) and not getattr(x, "_generated", False)][::-1]
	# 	print("proxyClases", proxyClasses)
	# 	cls._proxyParentCls = proxyClasses[-1]
	# 	cls._proxySuperCls = None
	# 	if len(proxyClasses) > 1:
	# 		cls._proxySuperCls = proxyClasses[-2]


	def __init__(self, obj, proxyData:ProxyData, **kwargs):
		"""called by __new__, _proxyLink is constructed internally"""
		#log("proxy init", vars=0)
		# dict for user data
		self._proxyData = proxyData
		# add to this dict instead of defining new attrs,
		# where possible - minimise impact on main Proxy class namespace


	@classmethod
	def _proxyObjUniqueId(cls, obj):
		"""return unique id to point to base object"""
		return id(obj)

	@classmethod
	def _existingProxy(cls, obj):
		"""retrieve an existing proxy object for the given base object"""
		return cls._objProxyCache.get(cls._proxyObjUniqueId(obj))

	# factories
	_special_names = [
		'__abs__', '__add__', '__and__', '__call__', '__cmp__', '__coerce__',
		'__contains__', '__delitem__', '__delslice__', '__div__', '__divmod__',
		'__eq__', '__float__', '__floordiv__', '__ge__', '__getitem__',
		'__getslice__', '__gt__', '__hash__', '__hex__', '__iadd__', '__iand__',
		'__idiv__', '__idivmod__', '__ifloordiv__', '__ilshift__', '__imod__',
		'__imul__', '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__',
		'__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__',
		'__long__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__',
		'__neg__', '__oct__', '__or__', '__pos__', '__pow__', '__radd__',
		'__rand__', '__rdiv__', '__rdivmod__', '__reduce__', '__reduce_ex__',
		'__repr__', '__reversed__', '__rfloorfiv__', '__rlshift__', '__rmod__',
		'__rmul__', '__ror__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__',
		'__rtruediv__', '__rxor__', '__setitem__', '__setslice__', '__sub__',
		'__truediv__', '__xor__', 'next',
	]
	_wrapTheseMethodsAnyway = (
		"__eq__", "__ge__", "__gt__", #"__hash__",
		"__le__", "__lt__", "__ne__",
	)

	def _beforeProxyCall(self, methodName:str,
	                     methodArgs:tuple, methodKwargs:dict,
	                     targetInstance:object
	                     )->tuple[T.Callable, tuple, dict, object]:
		"""overrides / filters args and kwargs passed to method
		getting _proxyBase and _proxyResult should both be legal here"""
		newMethod = getattr(targetInstance,methodName)
		#log("before proxy call", methodName, newMethod, methodArgs, methodKwargs, targetInstance)
		return newMethod, methodArgs, methodKwargs, targetInstance

	def _afterProxyCall(self, methodName:str,
	                    method:T.Callable,
	                     methodArgs:tuple, methodKwargs:dict,
	                    targetInstance: object,
	                    callResult:object,
	                    )->object:
		"""overrides / filters result of proxy method"""
		#log("after proxy call", methodName, method, methodArgs, methodKwargs, targetInstance, callResult)
		return callResult

	def _onProxyCallException(self,
	                          methodName: str,
	                          method:T.Callable,
	                          methodArgs: tuple, methodKwargs: dict,
	                          targetInstance: object,
	                          exception:BaseException)->(None, object):
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
		def _proxyMethod(self:Proxy, *args, **kw):
			# pre-call hook - must return parametres of method call
			newMethod, newArgs, newKw, targetInstance = self._beforeProxyCall(
				methodName, methodArgs=args, methodKwargs=kw,
				targetInstance=self._proxyResult()
			)

			try: # set up exception catch
				# actually call method on target instance, get raw result
				result = newMethod(*newArgs, **newKw)
			except Exception as e: # on any exception, pass to handler function
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
		# build namespace for generated class
		# combine declared proxy attributes
		allProxyAttrs = set(cls._proxyAttrs)
		for base in cls.__mro__:
			if getattr(base, "_proxyAttrs", None):
				allProxyAttrs.update(base._proxyAttrs)

		# work out parent, super and target classes
		mro = cls.__mro__
		proxyClasses = [x for x in cls.__mro__ if issubclass(x, Proxy) and not getattr(x, "_generated", False)][::-1]
		#print("proxyClases", cls, proxyClasses)
		_proxyParentCls = proxyClasses[-1]
		_proxySuperCls = None
		if len(proxyClasses) > 1:
			_proxySuperCls = proxyClasses[-2]

		# namespace dict
		namespace = {"_proxyAttrs" : allProxyAttrs,
		             "_generated" : True,
		             "_proxyTargetCls" : targetCls,
		             "_proxyParentCls" : _proxyParentCls,
		             "_proxySuperCls" : _proxySuperCls,
		             }
		toWrap = classCallables(targetCls)

		for methodName in toWrap:
			# do not override methods if they appear in proxy class
			# unless they are in the "wrapTheseMethodsAnyway" list
			#print("wrap", methodName, methodName in cls._wrapTheseMethodsAnyway, methodName in dir(cls))
			if hasattr(targetCls, methodName):
				if ((methodName in cls._wrapTheseMethodsAnyway)
						or (not methodName in dir(cls))):
					#print("wrapping", methodName)
					namespace[methodName] = cls._makeProxyMethod(methodName, targetCls)

		clsType = ProxyMeta

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
		newCls._proxyTargetCls = targetCls
		return newCls

	def __new__(cls, obj, *args, **kwargs):
		"""
        creates a proxy instance referencing `obj`. (obj, *args, **kwargs) are
        passed to this class' __init__, so deriving classes can define an
        __init__ method of their own.
        base Proxy class holds master dict

        proxying a proxy will create a new class for each layer
        and set of parents - this is not ideal, but hasn't hurt yet

        """

		# look up existing proxy classes
		cache = Proxy.__dict__["_classProxyCache"]
		#log("proxy new", cls, obj, vars=0)

		# check that we don't start generating classes from generated classes
		cls = next(filter(lambda x: not getattr(x, "_generated", False),
		                  cls.__mro__))

		# if obj is proxy, look at its type
		if Proxy in type(obj).__mro__:
			objCls = obj.__class__
		else:
			objCls = type(obj)
		try:
			genClass = cache[cls][objCls]
		except KeyError:
			genClass = cls._createClassProxy(objCls)
			cache[cls] = {objCls: genClass}

		# create new proxy instance of type-specific proxy class
		# sometimes gives "not safe" errors on builtin types
		#log("new proxy class", genClass, objCls, vars=0)
		try:
			ins = object.__new__(genClass)
		except TypeError:
			# for builtins need to call:
			#  int.__new__( ourNewClass, 3 )
			ins = objCls.__new__(genClass, obj)

		return ins

	@classmethod
	def getProxy(cls, targetObj,
	             shared=True,
	             proxyData=None,
	             proxyLinkCls=None,
	             **kwargs
	             ):
		"""preferred way of creating proxy - link class
		used falls back to class default,
		then to base ProxyLink if nothing else is specified

		if shared, an existing proxy for the given object will be returned
		(if available) - else, a new proxy object will be created

		if weak, no hard reference
		"""
		# check if shared proxy is available
		if isinstance(targetObj, Proxy):

			targetObj = targetObj._proxyStrongRef
		uniqueId = cls._proxyObjUniqueId(targetObj)
		if shared and uniqueId in cls._objProxyCache:
			if cls._objProxyCache[uniqueId]:
				print("fetching shared proxy reference for", uniqueId)
				return cls._objProxyCache[uniqueId]

		# retrieve class to use for ProxyLink
		#proxyLinkCls = proxyLinkCls or cls._proxyLinkCls or ProxyLink

		# generate ProxyLink component for this object

		#linkObj = proxyLinkCls(targetObj)
		proxyData = proxyData or ProxyData()
		proxyData["target"] = targetObj
		#print("proxyParentCls", cls._proxyParentCls, "proxySuperCls", cls._proxySuperCls)
		proxyObj : Proxy = cls.__new__(
			cls, targetObj,
		               proxyData=proxyData,
		               #proxyLink=linkObj,
			**kwargs
		               )
		proxyObj.__init__(targetObj, proxyData, **kwargs)
		#proxyObj._proxyStrongRef = targetObj
		cls._objProxyCache[uniqueId] = proxyObj

		#proxyObj._proxyStrongRef = targetObj
		return proxyObj

	def _setProxyBase(self, obj):
		"""set the base reference to the proxy"""
		self._proxyLink.setProxyTarget(obj)

	#@property
	def _proxyBase(self):
		return self._proxyLink.proxyTargetObj

	def _proxyResult(self):
		"""return the result of the _proxyLink process"""
		#return self._proxyLink.proxyResult()
		#return self._proxyLink.proxyResult()
		return self._proxyData["target"]


	def __getattr__(self, name):
		try: # look up attribute on proxy class first
			return object.__getattribute__(self, name)
		except:
			# obj = object.__getattribute__(self, self._proxyResultKey)
			obj = self._proxyResult()

			return getattr( obj, name)

	def __delattr__(self, name):
		# delattr(object.__getattribute__(self, self._proxyResultKey), name)
		delattr(self._proxyResult(), name)

	def __setattr__(self, name, value):
		try:
			if name in self.__pclass__._proxyAttrs:
				object.__setattr__(self, name, value)
			else:
				setattr(self._proxyResult(), name, value)
		except Exception as e:
			#print("p attrs {}".format(self.__pclass__._proxyAttrs))
			print("error name {}, value {}".format(name, value))
			print("type name ", type(name))
			raise e


	def __nonzero__(self):
		return bool(self._proxyResult())

	def __str__(self):
		return str(self._proxyResult())

	def __repr__(self):
		return repr(self._proxyResult())

	@property
	def __class__(self):
		"""used to spoof instance checks for proxy representation"""
		#print("request __class__ of ", self, self._proxyLink.baseClass)
		return type(self._proxyData["target"])

	@property
	def __pclass__(self):
		"""proxy class
		as __class__ is taken, this is to allow idoms like
		self.__class__.__name__ to continue with the minimum
		disruption
		also allows client code to use __class__ without triggering
		any proxy stuff, unless specifically working with it
		"""
		return type(self)

# Proxy._proxyParentCls = Proxy
# Proxy._proxySuperCls = None

class ProxyLink:
	"""smaller object to govern how the proxy resolves its base object
	initialised before proxy itself, reference to proxy set later
	"""
	def __init__(self, proxyTargetObj:object):
		#print("proxyLink init")
		self.baseClass = type(proxyTargetObj)
		self.proxyTargetObj = None
		self.setProxyTarget(proxyTargetObj)
		self.proxyInstance : Proxy = None # owning proxy object

	def setProxyInstance(self, proxy:Proxy):
		self.proxyInstance = proxy

	def setProxyTarget(self, proxyTargetObj):
		"""run once at setup"""
		self.proxyTargetObj = proxyTargetObj

	def getProxyTarget(self):
		"""in case something has to be run repeatedly to retrieve target object"""
		return self.proxyTargetObj

	def proxyResult(self):
		"""override this to do whatever you want
		return the target object to be used in place of the proxy -
		"""
		return self.getProxyTarget()

class LinkProxy(Proxy):
	"""
	same proxy logic, but we pass in a
	ProxyLink object to allow "repointing" this
	proxy at arbitrary target objects
	"""
	_classProxyCache = defaultdict(dict)  # { class : { class cache } }

	_generated = False
	_objProxyCache = {}  # NB: weakdict was giving issues, this might chug memory
	_proxyLinkCls = None  # optionally enforce a link class for this class of proxy
	_proxyAttrs = ("_proxyLink",
	               "_proxyData",
	               # "_proxyResult",
	               # "_proxyBase",
	               # "_proxyStrongRef"
	               )

	@classmethod
	def _proxyBaseCls(cls)->T.Type[Proxy]:
		"""return the base class for this proxy class -
		override if you subclass the 'master' class
		"""
		return LinkProxy

	def __new__(cls, obj, *args, **kwargs):
		"""
        creates a proxy instance referencing `obj`. (obj, *args, **kwargs) are
        passed to this class' __init__, so deriving classes can define an
        __init__ method of their own.
        base Proxy class holds master dict

        proxying a proxy will create a new class for each layer
        and set of parents - this is not ideal, but hasn't hurt yet

        """
		# print("proxy new", cls, obj, type(obj), )

		# look up existing proxy classes
		cache = Proxy.__dict__["_classProxyCache"]

		# check that we don't start generating classes from generated classes
		cls = next(filter(lambda x: not getattr(x, "_generated", False),
		                  cls.__mro__))

		# if obj is proxy, look at its type
		if Proxy in type(obj).__mro__:
			objCls = obj.__class__
		else:
			objCls = type(obj)
		try:
			genClass = cache[cls][objCls]
		except KeyError:
			genClass = cls._createClassProxy(objCls)
			cache[cls] = {objCls: genClass}

		# create new proxy instance of type-specific proxy class
		# sometimes gives "not safe" errors on builtin types
		try:
			ins = object.__new__(genClass)
		except TypeError:
			ins = objCls.__new__(genClass)

		return ins

	@classmethod
	def getProxy(cls, targetObj,
	             shared=True,
	             proxyData=None,
	             proxyLinkCls=None,
	             ):
		"""preferred way of creating proxy - link class
		used falls back to class default,
		then to base ProxyLink if nothing else is specified

		if shared, an existing proxy for the given object will be returned
		(if available) - else, a new proxy object will be created

		if weak, no hard reference
		"""
		# check if shared proxy is available
		if isinstance(targetObj, Proxy):
			targetObj = targetObj._proxyStrongRef
		uniqueId = cls._proxyObjUniqueId(targetObj)
		if shared and uniqueId in cls._objProxyCache:
			if cls._objProxyCache[uniqueId]:
				print("fetching shared proxy reference for", uniqueId)
				return cls._objProxyCache[uniqueId]

		# retrieve class to use for ProxyLink
		proxyLinkCls = proxyLinkCls or cls._proxyLinkCls or ProxyLink

		# generate ProxyLink component for this object

		linkObj = proxyLinkCls(targetObj)
		proxyObj = cls(targetObj,
		               proxyData=proxyData,
		               # proxyLink=linkObj,
		               )
		proxyObj._proxyStrongRef = targetObj
		cls._objProxyCache[uniqueId] = proxyObj

		proxyObj._proxyStrongRef = targetObj
		return proxyObj

	def __init__(self, obj, proxyData=None, **kwargs):
		proxySuper = inheritance.clsSuper(LinkProxy)
		proxySuper.__init__(self, obj, proxyData=proxyData, **kwargs)

		proxyLink = kwargs.get("proxyLink", None)
		assert proxyLink, "no proxy link provided for LinkProxy instance"
		self._proxyLink = proxyLink
		# after _proxyLink is set here, proxy is fully operational for rest of init

		# set the proxy link instance to this object
		self._proxyLink.setProxyInstance(self)



# class Proxy(Proxy # don't worry about it
#             ): # don't worry about it
# 	"""don't worry about it"""
# 	def __new__(cls, *args, **kwargs):
# 		oldNew = cls.__new__
# 		cls.__new__ = Proxy.__new__ # don't worry about it
# 		ins = cls.getProxy(*args)
# 		cls.__new__ = oldNew # don't worry about it
# 		return ins

# ^ no idea what this was about ^


if __name__ == '__main__':

	baseVal = 2
	proxVal = Proxy(2)
	print("prox val", proxVal)







