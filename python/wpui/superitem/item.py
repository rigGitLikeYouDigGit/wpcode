
from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from wplib.constant import LITERAL_TYPES, SEQ_TYPES, MAP_TYPES
from wplib.sentinel import Sentinel
from wplib.object import PluginRegister, PluginBase

from wpui.superitem.model import SuperModel
from wpui.superitem.view import SuperViewBase
from wpui.superitem.plugin import SuperItemPlugin
"""item -> model -> item"""




class SuperItem(QtGui.QStandardItem):
	"""base class for nested standarditems -
	use .forValue(), do not initialise directly
	"""

	pluginRegister = PluginRegister(
		systemName="superItem"
	)

	def __init__(self, _baseValue:T.Any=Sentinel.Empty):
		super(SuperItem, self).__init__()
		self.value = Sentinel.Empty
		self.childModel = SuperModel()

		if _baseValue is not Sentinel.Empty:
			self.setValue(_baseValue)

	def __repr__(self):
		return f"SuperItem({self.value})"

	@classmethod
	def forValue(cls, value)->SuperItem:
		itemCls = cls.getPlugin(value).itemCls or cls
		item = itemCls(value)
		return item


	@classmethod
	def getPlugin(cls, forValue)->type[SuperItemPlugin]:
		return cls.pluginRegister.getPlugin(type(forValue))

	@classmethod
	def registerPlugin(cls, plugin:type[PluginBase], forType):
		cls.pluginRegister.registerPlugin(plugin, forType)

	@classmethod
	def viewTypeForValue(cls, value)->type[SuperViewBase]:
		"""return a view type for a value -
		make this more extensible somehow"""
		return cls.getPlugin(value).viewCls

	def childIndexWidgetTypeMap(self)->dict[QtCore.QModelIndex, type[SuperViewBase]]:
		"""return map of model indices and view widgets for child items
		"""
		#print("indexMap")
		indexMap = {}
		for row in range(self.rowCount()):
			for column in range(self.columnCount()):
				item : SuperItem = self.item(row, column)
				#print("item", item)
				if item.hasChildModel():
					indexMap[item.index()] = item.viewTypeForValue(item.value)
		return indexMap


	def getNewView(self)->QtWidgets.QWidget:
		"""return a new view for this item
		break this out into a Policy object"""

		view = self.viewTypeForValue(self.value)()
		view.setModel(self.childModel)
		return view

	def createChildItem(self, value):
		return SuperItem(value)

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

	def setValue(self, value):
		self.childModel.clear()
		self.childModel = self.getPlugin(value).modelCls()
		self.value = value
		self.childModel.pyValue = value
		self.setUpChildModel(self.childModel, value)
		if isinstance(value, LITERAL_TYPES):
			self.setData(str(value), QtCore.Qt.DisplayRole)
			return
		rows = self.createOwnChildItems(value)
		#print("rows for", self, rows)

		for row in rows:
			self.childModel.appendRow(row)

	def setUpChildModel(self, childModel:QtGui.QStandardItemModel, value):
		if isinstance(value, MAP_TYPES):
			#headerModel = QtWidgets.QStringListModel()
			#headerModel.setStringList(["key", "value"])
			childModel.setHorizontalHeaderLabels(["key", "value"])



	def hasChildModel(self):
		return self.childModel.rowCount() > 0









