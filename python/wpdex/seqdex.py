


from __future__ import annotations
import typing as T

from wplib import log
from .base import WpDex

class SeqDex(WpDex):
	"""dict dex"""
	forTypes = (list, tuple)
	def _buildBranchMap(self, **kwargs) ->dict[DexPathable.keyT, WpDex]:
		#return [self.makeChildPathable((i,), v) for i, v in enumerate(self.obj)]
		#log("seqdex build children", vars=0)
		return {i : self._buildChildPathable(
			obj=v,
			#parent=self,
			name=i)
			for i, v in enumerate(self.obj)}
	def _consumeFirstPathTokens(self, path: pathT, **kwargs) ->tuple[list[WpDex], pathT]:
		"""process a path token
		:param **kwargs:
		"""
		token, *path = path
		# if isinstance(token, int):
		# 	return [self.children[token]], path
		return [self.branchMap()[token]], path
