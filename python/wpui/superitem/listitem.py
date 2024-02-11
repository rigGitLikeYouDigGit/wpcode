from __future__ import annotations

from wplib.constant import LITERAL_TYPES, SEQ_TYPES
from wpui.superitem import SuperModel, SuperItem


class ListSuperModel(SuperModel):
	"""model for a list"""
	forTypes = SEQ_TYPES


class ListSuperItem(SuperItem):
	"""superitem for a list"""
	forTypes = SEQ_TYPES


