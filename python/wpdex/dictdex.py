
from __future__ import annotations
import typing as T

from wplib import log
from .base import WpDex

class DictDex(WpDex):
	"""dict dex"""
	forTypes = (dict,)

	def validate(self):
		"""validate this object"""
		pass

	def _buildBranchMap(self):
		# return [(self.makeChildPathable(("keys()", ), k),
		#          self.makeChildPathable((k,), v) )
		#         for i, (k, v) in enumerate(self.obj.items())]
		#log("building children for dict", self.obj)
		items = { k : self._buildChildPathable(
			obj=v,
			#parent=self,
			name=k
		) for k, v in self.obj.items()}
		#items["keys()"] = self.makeChildPathable(("keys()",), list(self.obj.keys()))
		return items

	def _consumeFirstPathTokens(self, path: pathT, **kwargs) ->tuple[list[Pathable], pathT]:
		"""process a path token
		:param **kwargs:
		"""
		token, *path = path
		if token == "keys()":
			return [self.branchMap()["keys()"]], path
		return [self.branchMap()[token]], path



