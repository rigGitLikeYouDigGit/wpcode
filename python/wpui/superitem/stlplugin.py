
from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from wplib.constant import LITERAL_TYPES, SEQ_TYPES, MAP_TYPES
from wplib.sentinel import Sentinel
from wplib.object import PluginRegister, PluginBase

from wpui.model import iterAllItems

from wpui.superitem.base import SuperItem, SuperViewBase, SuperDelegate, SuperModel
from wpui.widget import WPTableView


"""what if whole thing in one"""

class SuperWidgetBase:
	parentSuperItem : SuperItem


class SuperListView(SuperViewBase, WPTableView):
	def __init__(self, *args, **kwargs):
		WPTableView.__init__(self, *args, **kwargs)
		SuperViewBase.__init__(self, *args, **kwargs)
		# corner = self.cornerWidget()
		# print(corner, type(corner))
		self.setCornerButtonEnabled(True)
		label = QtWidgets.QLabel("test")
		self.setCornerWidget(label)

class ListSuperItem(SuperItem):

	forCls = (list, tuple)
	viewCls = SuperListView

	def makeChildItems(self):
		for i in self.value:
			pluginItemCls = self.getPlugin(i)
			self.childModel.appendRow(
				pluginItemCls(i)
			)

	def getResultValue(self):
		return [i.getResultValue() for i in self.childSuperItems()]


class SuperDictModel(SuperModel):
	def headerData(self, section:int, orientation:QtCore.Qt.Orientation, role:int=...) -> T.Any:
		if role == QtCore.Qt.TextAlignmentRole:
			return QtCore.Qt.AlignLeft
		if role == QtCore.Qt.FontRole:
			f = QtGui.QFont()
			f.setPointSize(6)
			return f
		elif role == QtCore.Qt.DisplayRole:
			if orientation == QtCore.Qt.Horizontal:
				if section == 0:
					return "key"
				elif section == 1:
					return "value"
		return super(SuperModel, self).headerData(section, orientation, role)
	#


class SuperDictView(SuperViewBase, WPTableView):
	def __init__(self, *args, **kwargs):
		WPTableView.__init__(self, *args, **kwargs)
		SuperViewBase.__init__(self, *args, **kwargs)

		self.horizontalHeader().setStretchLastSection(True)
		self.horizontalHeader().setModel(
			QtCore.QStringListModel(["key", "value"])
		)
		self.horizontalHeader().model().setData(
			self.horizontalHeader().model().index(0, 0),
			"key",
			QtCore.Qt.DisplayRole

		)


class DictSuperItem(SuperItem):
	forCls = dict
	viewCls = SuperDictView
	modelCls = SuperDictModel

	def childSuperItems(self) ->list[tuple[SuperItem]]:
		items = []
		for i in range(self.childModel.rowCount()):
			key = self.childModel.item(i, 0)
			value = self.childModel.item(i, 1)
			items.append((key, value))
		return items

	def makeChildItems(self):
		for k, v in self.value.items():
			keyItem = self.getPlugin(k)(k)
			try:
				valueItem = self.getPlugin(v)(v)
			except TypeError:
				print("no plugin for", v, type(v))
			self.childModel.appendRow(
				[keyItem, valueItem]
			)

	def getResultValue(self):
		return {i[0].getResultValue():i[1].getResultValue()
		        for i in self.childSuperItems()}

# class LiteralView(QtWidgets.QLineEdit):
# 	pass

class LiteralDelegate(SuperDelegate):
	def paint(self, painter:PySide2.QtGui.QPainter, option:PySide2.QtWidgets.QStyleOptionViewItem, index:PySide2.QtCore.QModelIndex) -> None:
		rect = option.rect
		brush = QtGui.QBrush(QtGui.QColor(212, 255, 212))
		painter.save()
		painter.setBrush(brush)
		painter.drawRoundedRect(rect, 5, 5, mode=QtCore.Qt.AbsoluteSize)
		painter.restore()
		super(LiteralDelegate, self).paint(painter, option, index)
		# painter.drawText(option.rect, str(index.data()))

class LiteralSuperItem(SuperItem):
	"""For literal values, delegate to the delegate"""
	#viewCls = LiteralView
	forCls = LITERAL_TYPES
	delegateCls = LiteralDelegate

	def getNewView(self) ->viewCls:
		return None

	def getResultValue(self):
		return self.value

	def setValue(self, value):
		super(LiteralSuperItem, self).setValue(value)
		self.setData(value, role=QtCore.Qt.DisplayRole)

# class StringSuperItem(LiteralSuperItem):
# 	"""
# 	use a separate system for validation?
# 	"""
#
# 	forCls = str
#
# class FloatSuperItem(LiteralSuperItem):
# 	"""
# 	"""
#
# 	forCls = (int, float, complex)
#
# 	def setValue(self, value:str):
# 		super(FloatSuperItem, self).setValue(value)
# 		self.setData(value, role=QtCore.Qt.DisplayRole)


