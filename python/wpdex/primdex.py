
from __future__ import annotations
import typing as T


from .base import WpDex


# dexes for primitives
"""while we could easily have one dex type to take care of all
of these, it makes it easier in UI if we can specify which kind of primitive
is used here"""
class PrimDex(WpDex):
	#forTypes = (int, float, bool, type(None))

	def _buildBranchMap(self) ->dict[DexPathable.keyT, WpDex]:
		return {}

class IntDex(PrimDex):
	forTypes = (int, )
class BoolDex(PrimDex):
	forTypes = (bool, )
class NoneDex(PrimDex):
	forTypes = (type(None), )


