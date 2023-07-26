
from __future__ import annotations
import typing as T

class HashFunctions:
	"""for any builtin or restricted types that still need
	a custom function to return their hash"""
	typeFnMap : dict[type, T.Callable] = {}

	@classmethod
	def toHash(cls, obj)->int:
		try:
			return hash(obj)
		except:
			pass

		try:
			return cls.typeFnMap[type(obj)](obj)
		except KeyError:
			return id(obj)

toHash = HashFunctions.toHash

class HashableMixin:
	def __hash__(self):
		return id(self)

class HashableDict(dict):
	def __hash__(self):
		return hash(tuple(hash(i) for i in self.items()))

class UnHashableDict(dict):
	"""dict that accepts unhashable objects as keys"""

	def __init__(self):
		super(UnHashableDict, self).__init__()
		self.hashToObjectMap = {}  # map of hash to hashed key object

	def toHashValue(self, key)->int:
		return HashFunctions.toHash(key)

	def __setitem__(self, key, value):
		hashValue = self.toHashValue(key)
		self.hashToObjectMap[hashValue] = key
		super(UnHashableDict, self).__setitem__(hashValue, value)

	def __getitem__(self, item):
		hashValue = self.toHashValue(item)
		return super(UnHashableDict, self).__getitem__(hashValue)

	def __contains__(self, item):
		return super(UnHashableDict, self).__contains__(self.toHashValue(item))

	def get(self, key):
		return super(UnHashableDict, self).get(self.toHashValue(key))

	def keys(self):
		return [self.hashToObjectMap[i]
		        for i in super(UnHashableDict, self).keys()]


class UnHashableSet(set):
	"""compares objects by their id, not hash"""

	def __init__(self, seq=()):
		self._hashMap = {toHash(i) : i for i in seq}
		super(UnHashableSet, self).__init__(self._hashMap.keys())

	def add(self, element: _T) -> None:
		self._hashMap[toHash(element)] = element
	
	def __contains__(self, item):
		return super(UnHashableSet, self).__contains__(toHash(item))

	def intersection(self, other):
		return self.__class__(set(self).intersection(set(other)))

	def union(self, other):
		return self.__class__(set(self).union(set(other)))

	def difference(self, other):
		return self.__class__(set(self).difference(set(other)))

	def __sub__(self, other):
		return self.difference(other)

