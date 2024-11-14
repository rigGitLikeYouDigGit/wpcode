from __future__ import annotations

from wplib import Pathable
from .base import WpDex

class StrDex(WpDex):
	"""holds raw string
	may hold a child for the result of that string - uids,
	expressions, paths, etc
	"""
	forTypes = (str,)

	def _buildChildPathable(self) ->dict[DexPathable.keyT, WpDex]:
		"""build children"""
		return {}

	def bookendChars(self) ->tuple[str, str]:
		return "\"", "\""


class ExpDex(WpDex):
	"""holds final parsed expression
	this looms in a menacing fashion"""
	def _buildChildPathable(self) ->dict[DexPathable.keyT, WpDex]:
		"""build children"""
		return {}

