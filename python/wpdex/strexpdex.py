from __future__ import annotations

from wplib import Pathable
from .base import WpDex

class StrDex(WpDex):
	"""holds raw string
	may hold a child for the result of that string - uids,
	expressions, paths, etc
	"""
	forTypes = (str,)

	# def __init__(self, obj, parent, name):
	# 	super().__init__(obj, parent, name)
	# 	p = self.path
	# 	if None in self.path:
	# 		p = self.path
	# 	if "None" in self.path:
	# 		p = self.path

	# root dex has name of none
	# when is set as parent, child then has a parent,

	def _setParent(self, parent:Pathable):
		super()._setParent(parent)
		p = self.path
		if None in self.path:
			p = self.path
		if "None" in self.path:
			p = self.path
		pass


	def _buildChildPathable(self) ->dict[DexPathable.keyT, WpDex]:
		"""build children"""
		return {}
	pass


class ExpDex(WpDex):
	"""holds final parsed expression"""
	def _buildChildPathable(self) ->dict[DexPathable.keyT, WpDex]:
		"""build children"""
		return {}

