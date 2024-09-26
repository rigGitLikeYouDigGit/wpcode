
from __future__ import annotations
import typing as T


from .base import WpDex


# dexes for primitives

class PrimDex(WpDex):
	forTypes = (int, float, bool, type(None))

	def _buildBranchMap(self) ->dict[DexPathable.keyT, WpDex]:
		return {}



