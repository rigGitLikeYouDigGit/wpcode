
from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from wplib.constant import LITERAL_TYPES, SEQ_TYPES, MAP_TYPES
from wplib.sentinel import Sentinel

if T.TYPE_CHECKING:
	from wpui.superitem.item import SuperListView, SuperTableView, SuperItem


class SuperModel(QtGui.QStandardItemModel):
	"""not the photographic kind, this is way better"""

	def __init__(self, *args, **kwargs):
		super(SuperModel, self).__init__(*args, **kwargs)
		self.indexWidgetInstanceMap = {}


	def indicesForWidgets(self):
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

	def data(self, index:QtCore.QModelIndex, role:int=...) -> T.Any:
		if role == QtCore.Qt.TextAlignmentRole:
			return QtCore.Qt.AlignLeft
		return super(SuperModel, self).data(index, role)

	def headerData(self, section:int, orientation:QtCore.Qt.Orientation, role:int=...) -> T.Any:
		if role == QtCore.Qt.TextAlignmentRole:
			return QtCore.Qt.AlignLeft
		if role == QtCore.Qt.FontRole:
			f = QtGui.QFont()
			f.setPointSize(6)
			return f
		return super(SuperModel, self).headerData(section, orientation, role)


	# def headerData(self, section:int, orientation:PySide2.QtCore.Qt.Orientation, role:int=...) -> typing.Any:
	# 	"""return header data for section"""
	# 	print("header data", section, orientation, role, self.pyValue)
	# 	if isinstance(self.pyValue, MAP_TYPES):
	# 		labels = ["key", "value"]
	# 		if orientation == QtCore.Qt.Horizontal:
	# 			return labels[section]
	# 	else:
	# 		return str(section)