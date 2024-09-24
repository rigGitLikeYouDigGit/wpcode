
from __future__ import  annotations
import typing as T

from dataclasses import dataclass, asdict

from wplib.object import Adaptor

@dataclass#(frozen=True) # later on see if all deltas can be frozen
class DeltaAtom:
	"""do / undo a single atomic change on a target"""

	def do(self, target)->T.Any:
		"""apply delta to the target object and return it"""
		raise NotImplementedError
	def undo(self, target):
		"""remove delta from target object and return it"""
		raise NotImplementedError

	# def serialise(self):
	# 	return asdict(self)
	#
	# @classmethod
	# def fromDict(cls, dataDict:dict):
	# 	return cls(**dataDict)


# agnostic delta objects (should suffice for primitive types)
# we special-case string deltas later on?
primTypes = (tuple, list, set, dict)

@dataclass
class PrimDeltaAtom(DeltaAtom):
	"""container for delta functions on prim types"""

	@classmethod
	def _insertDict(cls, target:dict, key:str, index=None, value=None ):
		items = list(target.items())
		tie = (key, value)
		items.insert(index if index is not None else -1, tie)
		target.clear()  # preserve original object references
		target.update({**items})

	@classmethod
	def _insertTuple(cls, target:tuple, index:int, value)->tuple:
		l = list(target)
		l.insert(index, value)
		return type(target)(l)

	@classmethod
	def _popTuple(cls, target:tuple, index:int):
		l = list(target)
		value = l.pop(index)
		return type(target)(l), value

@dataclass
class InsertDelta(PrimDeltaAtom):
	"""insert value at key or index - signifies value has been created,
	with no prior source within delta scope"""
	key : str = None
	index : int = None
	value : object = None

	def do(self, target:primTypes):
		"""this could probably be more agnostic"""

		if isinstance(target, tuple):
			target = self._insertTuple(target, self.index, self.value)
		elif isinstance(target, T.Mapping):
			self._insertDict(target, self.key, self.index, self.value )
		elif self.index is None:  # we assume set?
			target.add(self.value)
		else:
			target.insert(self.index, self.value)
		super(InsertDelta, self).do(target)
		return target

	def undo(self, target:primTypes):
		if isinstance(target, tuple):
			target, value = self._popTuple(target, self.index)
		elif isinstance(target, T.Mapping):
			target.pop(self.key)
		elif self.index is None: # must be set
			target.remove(self.value)
		else:
			target.pop(self.index)
		super(InsertDelta, self).undo(target)
		return target

@dataclass
class RemoveDelta(InsertDelta):

	def do(self, target:primTypes):
		return super().undo(target)
	def undo(self, target:primTypes):
		return super().do(target)

@dataclass
class MoveDelta(PrimDeltaAtom):
	"""moving value between positions, both within delta scope-
	this delta doesn't even save the value itself"""
	oldKey : str = None
	oldIndex : int = None
	newKey : str = None
	newIndex : int = None

	def do(self, target:primTypes):
		if isinstance(target, tuple):
			target, value = self._popTuple(target, self.oldIndex)
			target = self._insertTuple(target, self.newIndex, value)
		elif isinstance(target, T.Mapping):
			self._insertDict(target, self.newKey,self.newIndex, target.pop(self.oldKey))
		else:
			target.insert(self.newIndex, target.pop(self.oldIndex))
		super(MoveDelta, self).do(target)
		return target

	def undo(self, target:primTypes):
		"""literally reverse of above"""
		if isinstance(target, tuple):
			target, value = self._popTuple(target, self.newIndex)
			target = self._insertTuple(target, self.oldIndex, value)
		elif isinstance(target, T.Mapping):
			self._insertDict(target, self.oldKey,self.oldIndex, target.pop(self.newKey))
		else:
			target.insert(self.oldIndex, target.pop(self.newIndex))
		super(MoveDelta, self).undo(target)
		return target



class DeltaAid(Adaptor):

	adaptorTypeMap = Adaptor.makeNewTypeMap()

	@classmethod
	def gatherDeltas(cls, baseObj, newObj)->list[DeltaAtom]:
		"""gather FROM base TO new"""
		raise NotImplementedError
