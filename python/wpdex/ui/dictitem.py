from __future__ import annotations

import typing as T

from wplib.constant import MAP_TYPES
from wplib.object import VisitAdaptor
from .base import WpDexModel, WpDexItem


class DictWpDexModel(WpDexModel):
	"""model for a dict"""
	forTypes = MAP_TYPES


class DictWpDexItem(WpDexItem):
	"""superitem for a dict"""
	forTypes = MAP_TYPES

	wpPyObj : dict

	@classmethod
	def getBookendChars(cls, forInstance=None) ->tuple[str, str]:
		"""return the characters to use as bookends for this item -
		"[", "]" for lists, "{", "}" for dicts, etc
		"""
		return ("{", "}")

	def _generateItemsFromPyObj(self) -> list[tuple[WpDexItem, VisitAdaptor.ChildType.T()]]:
		"""generate items from pyObj
		skip intermediate MapItem stage"""
		results = []
		for i in self.wpPyObj.items():
			keyItemType = self._getComponentTypeForObject(i[0], "item")
			valueItemType = self._getComponentTypeForObject(i[1], "item")
			results.extend((
				(keyItemType(
					i[0], wpChildType=VisitAdaptor.ChildType.MapKey,
				           parentQObj=self.wpChildModel,
				           parentSuperItem=self),
				 VisitAdaptor.ChildType.MapKey),
				(valueItemType(
					i[1], wpChildType=VisitAdaptor.ChildType.MapValue,
				           parentQObj=self.wpChildModel,
				           parentSuperItem=self),
				    VisitAdaptor.ChildType.MapValue) )
			)
		return results

	def wpResultObj(self) ->T.Any:
		"""return the result object"""
		results = []
		for i in range(0, len(self.wpChildSuperItems()), 2 ):
			key = self.wpChildSuperItems()[i].wpResultObj()
			value = self.wpChildSuperItems()[i+1].wpResultObj()
			results.append(((key, value), VisitAdaptor.ChildType.MapItem))

		return self.wpVisitAdaptor.newObj(
			self.wpPyObj,
			results
		)
