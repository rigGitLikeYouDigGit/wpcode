
from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from wplib.constant import LITERAL_TYPES, SEQ_TYPES, MAP_TYPES
from wplib.object import visit
from wplib.sentinel import Sentinel


"""item -> model -> item"""

base = object
if T.TYPE_CHECKING:
	base = QtWidgets.QAbstractItemView
# else:
# 	base = object


class SuperViewBase(
	base
                    ):

	sizeChanged = QtCore.Signal(QtCore.QSize)

	def __init__(self, *args, **kwargs):
		#super(SuperViewBase, self).__init__(*args, **kwargs)
		self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
		self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
		self.setAlternatingRowColors(True)
		self.setViewportMargins(0, 0, 0, 0)
		self.setContentsMargins(0, 0, 0, 0)
		# self.setSizePolicy(
		# 	QtWidgets.QSizePolicy.Fixed,
		# 	QtWidgets.QSizePolicy.Fixed,
		# )


	def syncSize(self):
		self.updateGeometries()
		self.update()
		self.updateGeometry()
		try:
			self.resizeRowsToContents()
		except:
			pass
		try:
			pass
			self.resizeColumnsToContents()
		except:
			pass
		try:
			self.setStretchLastSection(True)
		except:
			pass

		# try:
		# 	self.horizontalHeader().setStretchLastSection(True)

		# self.updateGeometries()
		# self.update()
		# self.updateGeometry()

	def resizeEvent(self, event):
		#self.sizeChanged.emit(self.size())
		for i in self.childViews():
			i.resizeEvent(event)
		self.syncSize()

		result = super(SuperViewBase, self).resizeEvent(event)
		return result

	def setModel(self, model:SuperModel):
		"""run over model to set view widgets on delegates"""
		delegate = SuperDelegate(self)
		self.setItemDelegate(delegate)
		super(SuperViewBase, self).setModel(model)

		indexMap = model.indicesForWidgets()
		instanceMap = {}
		for index, viewCls in indexMap.items():
			view = viewCls()
			instanceMap[index] = view
			item : SuperItem = model.itemFromIndex(index)
			view.setModel(item.childModel)
			self.setIndexWidget(index, view)
			item.childWidget = view

			delegate.sizeHintChanged.emit(index)
			view.syncSize()
		model.indexWidgetInstanceMap = instanceMap

		# # self.update()
		# #self.updateGeometries()
		# # self.updateGeometry()
		# #self.setRowHeight(0, 100)
		# #self.resize(self.sizeHint())
		self.syncSize()

		pass

	def childViews(self)->list[SuperViewBase]:
		"""return all child views"""
		return [
			self.indexWidget(index)
			for index in self.model().indicesForWidgets()
		]

	def onOuterWidgetSizeChanged(self):
		"""when outer widget size changes, we need to resize ourselves"""
		self.resize(self.sizeHint())
		for i in self.childViews():
			i.onOuterWidgetSizeChanged()

	def sizeHint(self):
		"""combine size hints of all rows and columns"""

		x = self.size().width()
		y = 0

		for i in range(self.model().rowCount()):
			y += self.sizeHintForRow(i) + 3

		try:
			y += self.horizontalHeader().height()
		except:
			pass
		#y += 20
		total = QtCore.QSize(x, y)
		return total

class SuperListView(SuperViewBase, QtWidgets.QListView):
	def __init__(self, *args, **kwargs):
		QtWidgets.QListView.__init__(self, *args, **kwargs)
		SuperViewBase.__init__(self, *args, **kwargs)

class SuperTableView(SuperViewBase, QtWidgets.QTableView):
	def __init__(self, *args, **kwargs):
		QtWidgets.QTableView.__init__(self, *args, **kwargs)
		SuperViewBase.__init__(self, *args, **kwargs)

		# header = self.horizontalHeader()
		#self.horizontalHeader().setFirstSectionMovable(False)
		self.horizontalHeader().setCascadingSectionResizes(True)
		self.horizontalHeader().setStretchLastSection(True)
		# self.setHorizontalHeader(header)

		# self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
		# self.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)



class SuperDelegate(QtWidgets.QStyledItemDelegate):
	"""delegate for superitem"""

	# def __init__(self, parent=None):
	# 	super(SuperDelegate, self).__init__(parent)

	def sizeHint(self, option:PySide2.QtWidgets.QStyleOptionViewItem, index:PySide2.QtCore.QModelIndex) -> PySide2.QtCore.QSize:
		"""return size hint for index - if complex, delegate to nested widget
		"""
		#print("size hint", index)
		if self.parent().indexWidget(index):
			item : SuperItem = self.parent().model().itemFromIndex(index)
			#print("size hint", item, item.childWidget.sizeHint())
			if item.childWidget:
				return item.childWidget.sizeHint()

		return super(SuperDelegate, self).sizeHint(option, index)

	# def paint(self, painter:PySide2.QtGui.QPainter, option:PySide2.QtWidgets.QStyleOptionViewItem, index:PySide2.QtCore.QModelIndex) -> None:
	# 	"""paint the item - if complex, delegate to nested widget
	# 	"""
	# 	#print("paint", index)
	# 	# if self.parent().indexWidget(index):
	# 	# 	#self.sizeHintChanged.emit(index)
	# 	# 	return self.parent().indexWidget(index).paint(painter, option, index)
	# 	return super(SuperDelegate, self).paint(painter, option, index)

class SuperModel(QtGui.QStandardItemModel):
	"""not the photographic kind, this is way better"""

	def __init__(self, *args, **kwargs):
		super(SuperModel, self).__init__(*args, **kwargs)
		self.indexWidgetInstanceMap = {}

	@classmethod
	def viewTypeForValue(cls, value):
		"""return a view type for a value"""
		if isinstance(value, SEQ_TYPES):
			return SuperListView
		elif isinstance(value, MAP_TYPES):
			return SuperTableView
		else:
			return None


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
					indexMap[item.index()] = self.viewTypeForValue(item.value)
		#print("indices for widgets", indexMap)
		#self.indexWidgetInstanceMap = indexMap
		return indexMap

	# def headerData(self, section:int, orientation:PySide2.QtCore.Qt.Orientation, role:int=...) -> typing.Any:
	# 	"""return header data for section"""
	# 	print("header data", section, orientation, role, self.pyValue)
	# 	if isinstance(self.pyValue, MAP_TYPES):
	# 		labels = ["key", "value"]
	# 		if orientation == QtCore.Qt.Horizontal:
	# 			return labels[section]
	# 	else:
	# 		return str(section)



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





