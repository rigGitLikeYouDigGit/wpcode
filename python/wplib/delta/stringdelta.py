
from __future__ import annotations
import typing as T

import difflib
from dataclasses import dataclass

from wplib.delta.abc import DeltaAtom, DeltaAid

@dataclass
class StringDeltaAtom(DeltaAtom):
	pass

class StringDeltaAid(DeltaAid):
	forTypes = (str, )

	@classmethod
	def gatherDeltas(cls, baseObj, newObj) ->list[DeltaAtom]:
		if baseObj == newObj:
			return []
		return super().gatherDeltas(baseObj, newObj)

