from __future__ import annotations

from wplib.constant import LITERAL_TYPES
from wpui.superitem import SuperItem


class LiteralSuperItem(SuperItem):
	"""model for a literal"""
	forTypes = LITERAL_TYPES
