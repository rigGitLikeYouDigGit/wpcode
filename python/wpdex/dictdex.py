
from __future__ import annotations
import typing as T

from wplib import log, Pathable
from .base import WpDex

class DictDex(WpDex):
	"""dict dex"""
	obj : dict
	forTypes = (dict,)

	def validate(self):
		"""validate this object"""
		pass

	def _buildBranchMap(self, **kwargs):
		# return [(self.makeChildPathable(("keys()", ), k),
		#          self.makeChildPathable((k,), v) )
		#         for i, (k, v) in enumerate(self.obj.items())]
		#log("building children for dict", self.obj)
		items = { k : self._buildChildPathable(
			obj=v,
			#parent=self,
			name=k
		) for k, v in self.obj.items()}
		for k, v in self.obj.items():
			keyName = f"key:{k}"
			items[keyName] = self._buildChildPathable(
				obj=k, name=keyName
			)
		return items

	def _consumeFirstPathTokens(self, path: pathT, **kwargs) ->tuple[list[Pathable], pathT]:
		"""process a path token
		:param **kwargs:
		"""
		token, *path = path
		log("dict consume first", token, path, self.branchMap())
		#log(self.branchMap())
		if token == "keys()":
			return [self.branchMap()["keys()"]], path
		return [self.branchMap()[token]], path

	def writeChildToKey(self, key:Pathable.keyT, value):
		"""this is why we wanted to have multiple tokens
		allowed for a single key
		nevermind, for now hack through"""
		if "key:" in str(key): # change one of the keys in the dict
			#strKeys = [f"key:{i}" for i in self.obj.keys()]
			strKeys = tuple(self.branchMap().keys())
			index = strKeys.index(key) // 2# + 1 # index of tie to modify
			log("key change", key, strKeys, index, value)
			items = [tuple(i) for i in self.obj.items()] # list of ties
			log("items", items) # object items not getting updated on creation?
			items[index] = (value, items[index][1]) # update single tie in list
			self.obj.clear() # keep reference to original value object
			self.obj.update({i[0] : i[1] for i in items}) # update with tie list
			return
		return super().writeChildToKey(key, value)

	def bookendChars(self)->tuple[str, str]:
		"""return 'bookends' that can be used to enclose a
		displayed version of this type"""
		return "{", "}"


