
from __future__ import annotations
import typing as T

from dataclasses import dataclass

from wplib.delta.abc import DeltaAtom, DeltaAid, MoveDelta, InsertDelta, RemoveDelta

"""super basic implementations for now, no attention to chunking, 
ordering of deltas, etc - 
 TODO: return to this
 
 
 current delta gathering looks at the whole object as an __eq__ - 
 no idea how that should be done
 """
class ListDeltaAid(DeltaAid):
	forTypes = (list, )

	@classmethod
	def gatherDeltas(cls, baseObj:list, newObj:list) ->list[DeltaAtom]:
		deltas = []
		toCheckNew = set(newObj)
		for i, val in enumerate(baseObj):
			if val not in toCheckNew:
				deltas.append(RemoveDelta(index=i, value=val))
				continue

			toCheckNew.remove(val)
			newIndex = newObj.index(val)
			if newIndex == i: continue
			deltas.append(MoveDelta(oldIndex=i, newIndex=newIndex))
		for newVal in toCheckNew:
			deltas.append(InsertDelta(
				newIndex=newObj.index(newVal),
				value=newVal
			))
		return deltas

class TupleDeltaAid(ListDeltaAid):
	forTypes = (tuple, )

@dataclass
class SetValueDelta(DeltaAtom):
	key : (int, str)
	oldVal : object
	newVal : object

	def do(self, target:dict) ->T.Any:
		target[self.key] = self.newVal
	def undo(self, target:dict):
		target[self.key] = self.oldVal


class PrimDeltaAid(DeltaAid):
	forTypes = (int, float, complex, bool)
	@classmethod
	def gatherDeltas(cls, baseObj, newObj) ->list[DeltaAtom]:
		if baseObj == newObj: return []
		return [{"change" : "value"}]


class DictDeltaAid(DeltaAid):
	forTypes = (dict, )
	@classmethod
	def gatherDeltas(cls, baseObj:dict, newObj:dict) ->list[DeltaAtom]:
		deltas = []
		toCheckNew = dict(newObj)
		oldKeys = tuple(baseObj.keys())
		newKeys = tuple(newObj.keys())
		for k, v in baseObj.items():
			if not k in newObj:
				deltas.append(RemoveDelta(key=k, value=v))
				continue
			toCheckNew.pop(k)
			if oldKeys.index(k) != newKeys.index(k):
				deltas.append(MoveDelta(oldKey=k, oldIndex=oldKeys.index(k),
				                        newKey=k, newIndex=newKeys.index(k)))
			if v != newObj[k]:
				deltas.append(SetValueDelta(k, oldVal=v, newVal=newObj[k]))

		for k, v in toCheckNew.items():
			deltas.append(InsertDelta(k, newKeys.index(k), value=v))
		return deltas


