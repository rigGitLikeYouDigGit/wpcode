
from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from wplib.constant import LITERAL_TYPES, SEQ_TYPES, MAP_TYPES
from wplib.sentinel import Sentinel
from wplib.object import PluginRegister, PluginBase

from wpui.model import iterAllItems

from wpui.superitem.base import SuperItem, SuperViewBase, SuperDelegate, SuperModel
from wpui.widget import WPTableView
from wpui.model import iterAllItems


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
		#self.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
		#self.setUniformRowHeights(False)


	def setModel(self, model: SuperModel):
		WPTableView.setModel(self, model)
		SuperViewBase.setModel(self, model)


	def sizeHintForRow(self, row:int) -> int:
		"""return size hint for row - if complex, delegate to nested widget"""

		#log("size hint for row", row)
		if not self.model():
			return super(SuperViewBase, self).sizeHintForRow(row)
		index = self.model().index(row, 0)
		if self.indexWidget(index):
			return self.indexWidget(index).sizeHint().height()
		return super(SuperViewBase, self).sizeHintForRow(row)


class ListSuperItem(SuperItem):

	forCls = (list, tuple)
	viewCls = SuperListView

	def makeChildItems(self):
		for i in self.superItemValue:
			pluginItem = self.forValue(i)
			self.childModel.appendRow(
				pluginItem
			)

	def getResultValue(self):
		return [i.getResultValue() for i in iterAllItems(model=self.childModel)]


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

		#self.horizontalHeader().setStretchLastSection(True)
		self.horizontalHeader().setModel(
			QtCore.QStringListModel(["key", "value"])
		)
		self.horizontalHeader().model().setData(
			self.horizontalHeader().model().index(0, 0),
			"key",
			QtCore.Qt.DisplayRole
		)
		# self.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)

		#self.setFixedSize(QtCore.QSize(200, 200))

	def setModel(self, model: SuperModel):
		WPTableView.setModel(self, model)
		SuperViewBase.setModel(self, model)

class DictSuperItem(SuperItem):
	forCls = dict
	viewCls = SuperDictView
	modelCls = SuperDictModel


	def makeChildItems(self):
		for k, v in self.superItemValue.items():
			keyItem = self.forValue(k)
			try:
				valueItem = self.forValue(v)
			except TypeError as e:
				print("no plugin for", v, type(v))
				raise e
			self.childModel.appendRow(
				[keyItem, valueItem]
			)

	def getResultValue(self):
		return {i[0].getResultValue():i[1].getResultValue()
		        for i in self.childSuperItems()}


class NoneSuperItem(SuperItem):
	forCls = type(None)
	viewCls = None

	def getResultValue(self):
		return None


# class LiteralView(QtWidgets.QLineEdit):
# 	pass

class LiteralDelegate(SuperDelegate):
	# def paint(self, painter:PySide2.QtGui.QPainter, option:PySide2.QtWidgets.QStyleOptionViewItem, index:PySide2.QtCore.QModelIndex) -> None:
	# 	rect = option.rect
	# 	brush = QtGui.QBrush(QtGui.QColor(212, 0, 0))
	# 	painter.save()
	# 	painter.setBrush(brush)
	# 	painter.drawRoundedRect(rect, 5, 5, mode=QtCore.Qt.AbsoluteSize)
	# 	painter.restore()
	# 	super(LiteralDelegate, self).paint(painter, option, index)
	# 	# painter.drawText(option.rect, str(index.data()))
	pass

class LiteralSuperItem(SuperItem):
	"""For literal values, delegate to the delegate"""
	viewCls = None
	forCls = LITERAL_TYPES
	delegateCls = LiteralDelegate
	#
	# def getNewView(self, parentQObject) ->viewCls:
	# 	return None

	def getResultValue(self):
		return self.superItemValue

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


