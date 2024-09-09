from . import DexPathable
from .base import WpDex

class StrDex(WpDex):
	"""holds raw string
	may hold a child for the result of that string - uids,
	expressions, paths, etc
	"""
	forTypes = (str,)

	def _buildChildren(self) ->dict[DexPathable.keyT, WpDex]:
		"""build children"""
		return {}
	pass


class ExpDex(WpDex):
	"""holds final parsed expression"""
	def _buildChildren(self) ->dict[DexPathable.keyT, WpDex]:
		"""build children"""
		return {}

