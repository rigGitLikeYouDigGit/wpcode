
from __future__ import annotations
import typing as T


VT = T.TypeVar("VT")
class SparseList(T.Generic[VT]):
	"""list that can have gaps in it"""
	__slots__ = ("_indexMap", )

	def __init__(self):
		self._indexMap : dict[int, VT] = {}

	@classmethod
	def fromItems(cls, items:dict[int, VT]):
		"""create a sparse list from a dict"""
		self = cls()
		self._indexMap = dict(items)
		return self

	@classmethod
	def fromTies(cls, ties:T.Iterable[T.Tuple[int, VT]]):
		"""create a sparse list from a list of tuples"""
		self = cls()
		self._indexMap = dict(ties)
		return self

	@classmethod
	def fromIndicesAndValues(cls, indices:list[int], values:list[VT]):
		"""create a sparse list from two lists"""
		self = cls()
		assert len(indices) == len(values), f"len(indices) != len(values) {len(indices)} != {len(values)}"
		self._indexMap = dict(zip(indices, values))
		return self

	def __getitem__(self, key:int)->VT:
		if key in self._indexMap:
			return self._indexMap[key]
		raise IndexError(f"Index {key} not in {self}")

	def __setitem__(self, key:int, value:VT):
		assert isinstance(key, int), f"key must be int, not {type(key)}"
		self._indexMap[key] = value

	def __repr__(self):
		return f'<{self.__class__.__name__}({self._indexMap})>'

	def __iter__(self)->T.Iterator[VT]:
		return iter(self._indexMap.values())

	def __len__(self):
		return len(self._indexMap)

	def enum(self)->list[T.Tuple[int, VT]]:
		return list(self._indexMap.items())

	def indices(self)->list[int]:
		return list(self._indexMap.keys())

	def fullEnum(self)->list[tuple[int, int, VT]]:
		"""return (rawIndex, sparseIndex, value)"""
		return list((i, k, v) for i, (k, v) in enumerate(self._indexMap.items()))


