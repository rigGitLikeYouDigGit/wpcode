
from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from wplib.constant import LITERAL_TYPES, SEQ_TYPES, MAP_TYPES
from wplib.sentinel import Sentinel
from wplib.object import PluginBase

from wpui.superitem.model import SuperModel
from wpui.superitem.delegate import SuperDelegate

if T.TYPE_CHECKING:
	from wpui.superitem import SuperItem
	from wpui.superitem.view import SuperViewBase

"""single class to override to add superItem support to arbitrary types
"""
#
class SuperItemPlugin(PluginBase):

	itemCls : type[SuperItem] = None
	viewCls : type[SuperViewBase] = None
	modelCls = SuperModel
	delegateCls = SuperDelegate

	def createOwnChildItems(self, value)->list[SuperItem]:
		rows = []
		if isinstance(value, MAP_TYPES):
			for i in value.items():
				rows.append(
					[self.createChildItem(s) for s in i]
					#[self.createChildItem(i[0]), self.createChildItem(i[1])]
				)
			return rows
		if isinstance(value, SEQ_TYPES):
			#rows.append(self.createChildItem(value))
			for i in value:
				rows.append(self.createChildItem(i))
			return rows
		raise TypeError(f"value {value} must be a sequence or mapping")

