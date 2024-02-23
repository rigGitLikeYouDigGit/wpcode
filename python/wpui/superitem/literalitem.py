from __future__ import annotations

import typing as T

from wplib.constant import LITERAL_TYPES
from wpui.superitem import SuperItem


class LiteralSuperItem(SuperItem):
	"""model for a literal"""
	forTypes = LITERAL_TYPES

	def wpResultObj(self) ->T.Any:
		"""return the result object"""
		return self.wpPyObj

class StringSuperItem(SuperItem):
	"""not sure how the split between string and literal might work
	"""
	forTypes = [str]

	def wpResultObj(self) ->T.Any:
		"""return the result object"""
		return self.wpPyObj
