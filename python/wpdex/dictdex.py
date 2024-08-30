
from __future__ import annotations
import typing as T


from .base import WpDex

class DictDex(WpDex):
	"""dict dex"""
	forTypes = (dict,)

	def validate(self):
		"""validate this object"""
		pass

	def _buildChildren(self):
		# return [(self.makeChildPathable(("keys()", ), k),
		#          self.makeChildPathable((k,), v) )
		#         for i, (k, v) in enumerate(self.obj.items())]
		items = { k : self.makeChildPathable((k,), v) for k, v in self.obj.items()}
		items["keys()"] = self.makeChildPathable(("keys()",), list(self.obj.keys()))
		return items

	def _consumeFirstPathTokens(self, path:pathT) ->tuple[list[Pathable], pathT]:
		"""process a path token"""
		token, *path = path
		if token == "keys()":
			return [self.children()["keys()"]], path
		return [self.children()[token]], path



