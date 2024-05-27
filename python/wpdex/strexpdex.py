

from .base import WpDex

class StrDex(WpDex):
	"""holds raw string
	may hold a child for the result of that string - uids,
	expressions, paths, etc
	"""
	forTypes = (str,)
	pass


class ExpDex(WpDex):
	"""holds final parsed expression"""

