
from __future__ import annotations
import typing as T


from .base import WpDex, DexPathable


# dexes for primitives

class PrimDex(WpDex):
	forTypes = (int, float, bool, None)

	def _buildChildren(self) ->dict[DexPathable.keyT, WpDex]:
		return {}



