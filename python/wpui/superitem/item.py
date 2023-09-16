
from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from wplib.constant import LITERAL_TYPES, SEQ_TYPES, MAP_TYPES
from wplib.sentinel import Sentinel

from wpui.superitem.model import SuperModel
from wpui.superitem.view import SuperViewBase, SuperListView, SuperTableView

"""item -> model -> item"""




class SuperItem(QtGui.QStandardItem):
	"""base class for nested standarditems - """

	def __init__(self, baseValue:T.Any=Sentinel.Empty):
		super(SuperItem, self).__init__()
		self.value = Sentinel.Empty
		self.childModel = SuperModel()

		# I know this is mixing up MVC in ways that should never be done,
		# but it's only within the domain of this widget system
		self.childWidget :SuperViewBase = None


		if baseValue is not Sentinel.Empty:
			self.setValue(baseValue)

	def __repr__(self):
		return f"SuperItem({self.value})"

	@classmethod
	def viewTypeForValue(cls, value):
		"""return a view type for a value -
		make this more extensible somehow"""
		if isinstance(value, SEQ_TYPES):
			return SuperListView
		elif isinstance(value, MAP_TYPES):
			return SuperTableView
		else:
			return None

	def getNewView(self)->QtWidgets.QWidget:
		"""return a new view for this item
		break this out into a Policy object"""
		view = None
		if isinstance(self.value, SEQ_TYPES):
			view = SuperListView()
		elif isinstance(self.value, MAP_TYPES):
			view = SuperTableView()

		if view is None:
			return None

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



if __name__ == '__main__':
	import sys

	structure = {
		"root": {
			"a": 1,

			"b": (2, 3),
			# "branch": {
			# 	(2, 4) : [1, {"key" : "val"}, "chips", 3],
			# }
		},
		"root2": {
			"a": 1,
		}
	}

	app = QtWidgets.QApplication([])

	item = SuperItem(structure)
	w = item.getNewView()


	w.show()
	sys.exit(app.exec_())





