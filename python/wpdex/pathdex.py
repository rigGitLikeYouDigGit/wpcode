from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
import pathlib
from .base import WpDex

from wplib import Pathable

# dex for paths

class PathDex(WpDex):

	forTypes = (pathlib.PurePath, )

	def _buildBranchMap(self, **kwargs) ->dict[DexPathable.keyT, WpDex]:
		return {}
		return {i: self._buildChildPathable(
			part, i
		) for i, part in enumerate(self.obj.parts)}

class PathableDex(WpDex):

	forTypes = (Pathable, )

	def _buildBranchMap(self, **kwargs) ->dict[DexPathable.keyT, WpDex]:
		return {k: self._buildChildPathable(v, k)
		        for k, v in self.obj._buildBranchMap().items()}


