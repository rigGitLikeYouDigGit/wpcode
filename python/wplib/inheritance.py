from __future__ import annotations, print_function

import types
import typing as T
import inspect
from ordered_set import OrderedSet

from wplib import log
from wplib.sentinel import Sentinel

# function to detect namedtuple instances - can be annoying otherwise
# NB: namedtuples also can't initialise around sequences, like normal tuples can
def isNamedTupleInstance(obj:T.Any)->bool:
	"""detect if obj is a namedtuple instance"""
	return isinstance(obj, tuple) and hasattr(obj, "_fields")

def isNamedTupleClass(cls:T.Any)->bool:
	"""detect if cls is a namedtuple class"""
	return issubclass(cls, tuple) and hasattr(cls, "_fields")

def clsSuper(cls:type)->type:
	"""return the superclass of cls"""
	return cls.__mro__[1]

class RichNotImplementedError(NotImplementedError):
	"""subclass of NotImplementedError, giving more description
	on exactly where it was raised, and from which type

	... is there a reason to not just pass NotImplementedError( self )
	and print that?

	if it's not ridiculously overcomplicated,
	how am I supposed to validate myself
	"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.callFrame = inspect.currentframe().f_back


def leafParentBases(*desiredBases:tuple[type])->tuple[type, ...]:
	"""given selection of superclasses to inherit from,
	return the actual bases to pass, to generate working mro

	eg if one is a subclass of the other, only pass the lowest base

	super temp for now, come back and make this work properly if needed
	"""
	#resultBases = {desiredBases[0]}
	resultBases = OrderedSet((desiredBases[0], ))
	for secondaryBase in desiredBases[1:]:
		resultBases -= set(secondaryBase.__mro__)
		resultBases.add(secondaryBase.__mro__[0])
	return tuple(resultBases)

def resolveInheritedMetaClass(*realBases:T.Sequence[type])->type[type]:
	"""given a list of "real" parent classes to inherit
	from, resolve a metaclass that inherits from any
	metaclasses defined in those bases"""
	# only get custom defined metaclasses
	metaBases = [type(i) for i in realBases if i is not type]
	if not metaBases: return type # no metaclass at all

	metaLeaves = leafParentBases(*metaBases)
	genName = f"GEN_META({metaLeaves})"
	return type(genName, metaLeaves, {})



def containsSuperClass(classSeq:T.Sequence[type], lookup:(type, object))->(type, None):
	"""returns first item in sequence that is a superclass of lookup,
	sort sequence externally to put most specific classes first"""
	if not isinstance(lookup, type):
		lookup = type(lookup)
	for i in classSeq:
		if not isinstance(i, type):
			continue
		#if i in lookup.__mro__:
		if issubclass(lookup, i): # really hope this works
			return i
	return None

def superClassLookup(classMap:(dict[type], dict[tuple[type]]), lookupCls:(type, object), default=None):
	"""indexes in to a map of {type : value} using lookupCls"""
	matching = containsSuperClass(classMap, lookupCls)
	if matching is None: return default
	return classMap[matching]


class SuperClassLookupMap:
	"""wrapping the above in a class for more consistent
	use and caching

	can maybe pass this as param to class initialisation,
	as well as defining at class scope, to set different type
	overrides for different parts of the program
	"""
	def __init__(self, classMap:dict[type, T.Any]=None):
		self.classMap : dict[type, T.Any] = {}
		self.cacheMap : dict[type, T.Any] = {}
		self._hasMatchFunctions = False
		if classMap is not None:
			self.updateClassMap(classMap)

	def __repr__(self):
		return f"{self.__class__}({self.classMap})"

	def copy(self):
		"""return a copy of this object"""
		return SuperClassLookupMap(self.classMap)

	def _expandTypeTupleKeys(self, classMap:dict[type, T.Any]):
		"""expand out any tuple keys and then sort"""
		testMap = dict(classMap or {})
		resultMap = {}
		for k, v in testMap.items():
			if isinstance(k, tuple):
				for i in k:
					resultMap[i] = v
			else:
				resultMap[k] = v
		return resultMap

	def _sortMap(self):
		"""sort the map by length of mro,
		so that longest mro (lowest superclasses) are first"""
		self.classMap = dict(
			sorted(self.classMap.items(),
			       key=lambda i: len(i[0].__mro__) if isinstance(i[0], type) else 0,
			       reverse=True)
		)
		for i in self.classMap.keys():
			if isinstance(i, types.FunctionType):
				self._hasMatchFunctions = True

	def updateClassMap(self, classMap:dict[(type, tuple[type, ...]), T.Any]):
		"""register a map of {type : value}"""
		#log("updateClassMap", classMap)
		self.classMap.update(self._expandTypeTupleKeys(classMap))
		self._sortMap()
		self.cacheMap.clear()
		#log("end hasfn", self._hasMatchFunctions)

	def lookup(self, lookupCls:type, default=Sentinel.FailToFind):
		"""lookup a value using lookupCls,
		add found results to cache.

		Explicitly specify default=None to pass None from failure,
		otherwise it will raise a KeyError
		"""
		#log(f"lookup {lookupCls} in {self.classMap}")
		if lookupCls in self.cacheMap:
			return self.cacheMap[lookupCls]
		if lookupCls is None: # None messes up everything
			result = self.classMap.get(None, default=Sentinel.FailToFind)
		else:
			result = superClassLookup(
				self.classMap, lookupCls, default=Sentinel.FailToFind)
		#log(f"result {result}")
		if result is Sentinel.FailToFind: # not found in map
			if self._hasMatchFunctions: # check against functions now
				for k, v in self.classMap.items():
					#log("check", k, v)
					if isinstance(k, types.FunctionType):
						if k(lookupCls):
							result = v

		if result is not Sentinel.FailToFind: # found in map
			self.cacheMap[lookupCls] = result
			return result

		if default is Sentinel.FailToFind :
			raise KeyError(f"No value registered in {self.classMap}\n\nfor {lookupCls}")
		return default



def iterSubClasses(cls, _seen=None, includeTopCls=False)->T.Generator[T.Type["cls"]]:
	"""
	iterSubClasses(cls)
	http://code.activestate.com/recipes/576949-find-all-subclasses-of-a-given-class/
	Generator over all subclasses of a given class, in depth first order.
	"""

	if not isinstance(cls, type):
		raise TypeError('iterSubClasses must be called with '
						'new-style classesToReload, not %.100r' % cls)
	if _seen is None: _seen = set()
	try:
		subs = cls.__subclasses__()
	except TypeError:  # fails only when cls is type
		subs = cls.__subclasses__(cls)
	if includeTopCls:
		subs = [cls] + list(subs)
	for sub in subs:
		if sub not in _seen:
			_seen.add(sub)
			yield sub
			for sub in iterSubClasses(sub, _seen, includeTopCls=False):
				yield sub

def mroMergedDict(cls):
	"""returns a merged dict of all superclasses of cls,
	matching override order during inheritance"""
	merged = {}
	for i in cls.__mro__:
		merged.update(i.__dict__)
	return merged

def classDescriptors(cls, inherited=True)->dict[str, T.Any]:
	"""return all objects of this class that define
	__get__, __set__, or __del__ """
	# descriptors are instances - need to check the type of the attribute of the class
	return {k : v for k, v in mroMergedDict(cls).items()
	        if hasattr(type(v), "__dict__") and any(
		i in type(v).__dict__ for i in ("__get__", "__set__")
		)
	        }


# annotation decorators
def overrideThis(fn:T.Callable)->T.Callable:
	"""decorator to mark a method as needing to be overridden
	they actually added something like this as a PEP, maybe I'm not
	insane after all
	"""
	return fn


# declare method as classmethod, unless called on a class instance
# pattern is to have classes and class methods as general practice,
# but then to override special cases with specific instances.

def classCallables(cls, dunders=True):
	"""return all attributes of a class that can be called"""
	methodNames = []
	mroDict = mroMergedDict(cls)
	for attrName in dir(cls):
		# this fails on descriptors, since it invokes them
		#attrValue = getattr(cls, attrName)
		attrValue = mroDict[attrName]
		if callable(attrValue):
			if any((dunders, not attrName.startswith("__"))):
				methodNames.append(attrName)
	return methodNames

if __name__ == '__main__':

	class Base:
		at = 3
		baseAt = "ey"

	class Leaf(Base):
		at = 5

	log(Leaf.__dict__)
	log(mroMergedDict(Leaf))
	log(mroMergedDict(Leaf)["baseAt"])



