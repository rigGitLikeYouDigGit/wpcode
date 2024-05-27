


from __future__ import annotations
import typing as T

from . import DexPathable
from .base import WpDex

class SeqDex(WpDex):
	"""dict dex"""
	forTypes = (list, tuple)
	def _buildChildren(self) ->dict[DexPathable.keyT, WpDex]:
		#return [self.makeChildPathable((i,), v) for i, v in enumerate(self.obj)]
		return {i : self.makeChildPathable((i,), v) for i, v in enumerate(self.obj)}
	def _consumeFirstPathTokens(self, path:pathT) ->tuple[list[WpDex], pathT]:
		"""process a path token"""
		token, *path = path
		# if isinstance(token, int):
		# 	return [self.children[token]], path
		return [self.children[token]], path
