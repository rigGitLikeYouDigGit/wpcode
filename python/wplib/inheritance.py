from __future__ import annotations, print_function

import typing as T

from wplib import log
from wplib.sentinel import Sentinel


def leafParentBases(*desiredBases:tuple[type])->list[type]:
	"""given selection of superclasses to inherit from,
	return the actual bases to pass, to generate working mro

	eg if one is a subclass of the other, only pass the lowest base

	super temp for now, come back and make this work properly if needed
	"""
	#mainSeq = set(desiredBases[0].__mro__)
	resultBases = {desiredBases[0]}
	for secondaryBase in desiredBases[1:]:
		resultBases -= set(secondaryBase.__mro__)
		resultBases.add(secondaryBase.__mro__[0])
	return resultBases


def containsSuperClass(classSeq:T.Sequence[type], lookup:(type, object))->(type, None):
	"""returns any items in the sequence which are superclasses of lookup"""
	if not isinstance(lookup, type):
		lookup = type(lookup)
	for i in classSeq:
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
	"""
	def __init__(self, classMap:dict[type, T.Any]=None):
		self.classMap : dict[type, T.Any] = {}
		self.cacheMap : dict[type, T.Any] = {}
		if classMap is not None:
			self.updateClassMap(classMap)

	def _expandTupleKeys(self, classMap:dict[type, T.Any]):
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
			       key=lambda i: len(i[0].__mro__),
			       reverse=True)
		)

	def updateClassMap(self, classMap:dict[type, T.Any]):
		"""register a map of {type : value}"""
		self.classMap.update(self._expandTupleKeys(classMap))
		self._sortMap()
		self.cacheMap.clear()

	def lookup(self, lookupCls:type, default=Sentinel.FailToFind):
		"""lookup a value using lookupCls,
		add found results to cache"""
		#log(f"lookup {lookupCls} in {self.classMap}")
		if lookupCls in self.cacheMap:
			return self.cacheMap[lookupCls]
		result = superClassLookup(
			self.classMap, lookupCls, default=Sentinel.FailToFind)
		#log(f"result {result}")
		if result is not Sentinel.FailToFind: # found in map
			self.cacheMap[lookupCls] = result
			return result
		if default is Sentinel.FailToFind :
			raise KeyError(f"No value registered in {self.classMap}\n"
			               f" for {lookupCls}")
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

# annotation decorators
def overrideThis(fn:T.Callable)->T.Callable:
	"""decorator to mark a method as needing to be overridden
	"""
	return fn