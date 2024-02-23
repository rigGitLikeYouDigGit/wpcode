from __future__ import annotations

import typing as T

from wplib.constant import LITERAL_TYPES, SEQ_TYPES
from wpui.superitem import SuperModel, SuperItem


class ListSuperModel(SuperModel):
	"""model for a list"""
	forTypes = SEQ_TYPES


class ListSuperItem(SuperItem):
	"""superitem for a list"""
	forTypes = SEQ_TYPES

	def wpResultObj(self) ->T.Any:
		"""return the result object"""
		return self.wpVisitAdaptor.newObj(
			self.wpPyObj,
			((i.wpResultObj(), i.wpChildType) for i in self.wpChildSuperItems())
		)
