


from __future__ import annotations
import pprint, copy, textwrap, ast
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from wplib.constant import LITERAL_TYPES, SEQ_TYPES, MAP_TYPES
from wplib.sentinel import Sentinel
from wplib.object import PluginRegister, PluginBase
from wplib import log

from wpui.model import iterAllItems

from wpui.widget import WPTableView

"""what if whole thing in one"""


class SuperDelegate(QtWidgets.QStyledItemDelegate):
	"""delegate for superitem

	It seems if you set an indexWidget for a view index, the delegate is not called
	at all.

	We need to run all size hint stuff inside delegates, because the view
	the view only defines sizeHintForRow, sizeHintForColumn,
	which don't work for nested items

	this WORKS, as long as embedded widgets report their sizeHint correctly
	if you set it manually it's great
	focus now on getting sizehint to trigger properly

	"""

	# def __init__(self, parent=None):
	# # 	super(SuperDelegate, self).__init__(parent)
	#
	def _sizeHintForIndex(self, option:QtWidgets.QStyleOptionViewItem, index:QtCore.QModelIndex) -> PySide2.QtCore.QSize:
		"""return size for single index"""
		if self.parent().indexWidget(index):
			return self.parent().indexWidget(index).sizeHint()
		return super(SuperDelegate, self).sizeHint(option, index)
		pass
	def sizeHint(self, option:PySide2.QtWidgets.QStyleOptionViewItem, index:PySide2.QtCore.QModelIndex) -> PySide2.QtCore.QSize:
		"""return size hint for index - if complex, delegate to nested widget
		"""
		#log("size hint for:", index.model().itemFromIndex(index), self.parent().indexWidget(index))
		#return QtCore.QSize(100, 100)

		item = index.model().itemFromIndex(index)

		# return maximum of all items in row
		if index.parent().isValid():
			rowItems = index.model().columnCount(index.parent())
		else:
			rowItems = index.model().columnCount()
		baseSize = self._sizeHintForIndex(option, index)
		height = baseSize.height()
		height = 500
		width = baseSize.width()
		#width = 50
		#log("rowItems", index, rowItems)
		for i in range(rowItems):
			#if index.parent().isValid():
			childIndex = index.model().index(index.row(), i, index.parent())
			#childIndex = index.parent().child(index.row(), i)
			if not childIndex.isValid():
				#log(item, "invalid child index", childIndex)
				continue

			indexSize = self._sizeHintForIndex(option, childIndex)
			#log("item", item, indexSize, indexSize.height(), "for", childIndex.model().itemFromIndex(childIndex))
			height = max(height, indexSize.height())
			#width += indexSize.width()
		#width = self._sizeHintForIndex(option, index).width()

		#log("endSizeHint for", item, height, width, "rowItems", rowItems)
		return QtCore.QSize(width, height)
		# if self.parent().indexWidget(index):
		# 	return self.parent().indexWidget(index).sizeHint()
		# 	item : SuperItem = self.parent().model().itemFromIndex(index)
		# 	#print("size hint", item, item.childWidget.sizeHint())
		# 	if item.childWidget:
		# 		return item.childWidget.sizeHint()

		return super(SuperDelegate, self).sizeHint(option, index)


class SuperModel(QtGui.QStandardItemModel):
	"""not the photographic kind, this is way better"""

	def __init__(self):
		super(SuperModel, self).__init__()
		self.parentSuperItem : SuperItem = None

	def data(self, index:QtCore.QModelIndex, role:int=...) -> T.Any:
		if role == QtCore.Qt.TextAlignmentRole:
			return QtCore.Qt.AlignLeft
		#log("data", index, role, self.itemFromIndex(index))
		return super(SuperModel, self).data(index, role)

	def headerData(self, section:int, orientation:QtCore.Qt.Orientation, role:int=...) -> T.Any:
		if role == QtCore.Qt.TextAlignmentRole:
			return QtCore.Qt.AlignLeft
		if role == QtCore.Qt.FontRole:
			f = QtGui.QFont()
			f.setPointSize(6)
			return f
		return super(SuperModel, self).headerData(section, orientation, role)
	#




base = object
if T.TYPE_CHECKING:
	base = QtWidgets.WPTableView
# else:
# 	base = object



class SuperViewBase(
	base
                    ):
	parentSuperItem : SuperItem

	sizeChanged = QtCore.Signal(QtCore.QSize)

	def __init__(self, *args, **kwargs):
		#super(SuperViewBase, self).__init__(*args, **kwargs)
		self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
		self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
		self.setAlternatingRowColors(True)
		self.setViewportMargins(0, 0, 0, 0)
		self.setContentsMargins(0, 0, 0, 0)
		#self.setFrameShape(QtWidgets.QFrame.NoFrame)
		self.setFrameShape(QtWidgets.QFrame.StyledPanel)

		#self.setFixedSize(400, 400) # THIS WORKS
		self.setItemDelegate(SuperDelegate(self))

		self.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
		#self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

		#self.setWidgetResizable(True)

	if T.TYPE_CHECKING:
		def model(self)->SuperModel:
			pass

	# for some reason this is never called
	# def sizeHintForIndex(self, index:QtCore.QModelIndex) -> QtCore.QSize:
	# 	"""return size hint for index - if complex, delegate to nested widget"""
	# 	log("size hint for index", index, self.indexWidget(index))
	# 	if self.indexWidget(index):
	# 		return self.indexWidget(index).sizeHint()
	# 	return super(SuperViewBase, self).sizeHintForIndex(index)
	#
	# def sizeHintForRow(self, row:int) -> int:
	# 	"""return size hint for row - if complex, delegate to nested widget"""
	# 	#log("size hint for row", row)
	# 	index = self.model().index(row, 0)
	# 	if self.indexWidget(index):
	# 		return self.indexWidget(index).sizeHint()
	# 	return super(SuperViewBase, self).sizeHintForRow(row)

	# def sizeHintForIndex(self, index:QtCore.QModelIndex) -> QtCore.QSize:
	# 	"""return size hint for index - if complex, delegate to nested widget"""
	# 	#log("size hint for index", index, self.indexWidget(index))
	# 	if self.indexWidget(index):
	# 		return self.indexWidget(index).sizeHint()
	# 	return super(SuperViewBase, self).sizeHintForIndex(index)

	def sizeHintForSingleRow(self, index,one):
		rowCount = self.model().rowCount(index)
		width = 0
		height = 0
		for i in range(rowCount):
			indexSize = self.sizeHintForIndex(self.model().index(i, 0, index))
			height = max(height, indexSize.height())
			width += indexSize.width()
		return QtCore.QSize(width, height)


	def sizeHint(self:QtWidgets.QAbstractItemView):
		"""combine size hints of all rows and columns"""
		#log("base view sizehint", self)
		return QtCore.QSize(200, 200)
	# 	#return self.contentsRect().size()
	# 	x = self.size().width()
	# 	y = 0
	#
	# 	# for item in iterAllItems(model=self.model()):
	# 	# 	y += self.sizeHintForIndex(item.index()).height()
	# 	for i in range(self.model().rowCount()):
	# 		y += self.sizeHintForRow(i) + 2
	#
	# 	try:
	# 		y += self.horizontalHeader().height()
	# 	except:
	# 		pass
	# 	#y += 20
	# 	total = QtCore.QSize(x, y)
	# 	return total

	def onItemChanged(self, item:QtGui.QStandardItem):
		"""item changed - update view"""
		self.regenWidgets()
		#self.syncSize()
		pass

	def childWidgets(self:QtWidgets.QAbstractItemView)->T.Iterable[QtWidgets.QWidget]:
		"""return all child widgets of view"""
		for obj in self.children():
			if isinstance(obj, SuperViewBase):
				#print("child widget", obj)
				yield obj

	def syncChildWidgetSizes(self:QtWidgets.QAbstractItemView):
		"""sync child widget sizes"""
		for childWidget in self.childWidgets():
			childWidget.syncChildWidgetSizes()
		self.updateGeometries()
		self.updateGeometry()
		self.setMinimumSize(self.sizeHint())

	def regenWidgets(self):
		#log("regen widgets", self, self.model(), self.parentSuperItem, self.parentSuperItem.childModel)
		# if self.model() is None:
		# 	print("no model for ", self)
		# 	return
		for item in iterAllItems(model=self.model()):
			if not isinstance(item, SuperItem):
				continue
			if item.childModel is not None:
				w :QtWidgets.QAbstractItemView = item.getNewView(parentQObject=self)
				if w is None:
					continue
				self.setIndexWidget(item.index(), w)
				#w.setMinimumSize(w.sizeHint())
				#w.setFixedSize(QtCore.QSize(100, 100))


	def setModel(self, model: SuperModel):
		"""run over model to set view widgets on delegates"""

		model.itemChanged.connect(self.onItemChanged)
		self.regenWidgets()
		self.syncChildWidgetSizes()
		#self.syncSize()

		pass


	def syncSize(self):
		#return
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




	# def parentSuperItem(self)->SuperItemBase:
	# 	return self.model().parentSuperItem

	def childViews(self)->list[SuperViewBase]:
		"""return all child views"""
		return [
			self.indexWidget(index)
			for index in self.model().indicesForWidgets()
		]

class SuperItem(QtGui.QStandardItem, PluginBase):
	"""inheriting from standardItem for the master object,
	since that lets us reuse the model/item link
	to map out structure.
	"""

	forCls = None
	pluginRegister = PluginRegister("superItemClasses")

	@classmethod
	def getPlugin(cls, forValue)->type[SuperItem]:
		return cls.pluginRegister.getPlugin(type(forValue))

	@classmethod
	def registerPlugin(cls, plugin:type[SuperItem], forType):
		cls.pluginRegister.registerPlugin(plugin, forType)


	modelCls = SuperModel
	viewCls = SuperViewBase
	#itemCls = QtGui.QStandardItem
	delegateCls = QtWidgets.QStyledItemDelegate

	def __init__(self):
		super(SuperItem, self).__init__()
		self.superItemValue = Sentinel.Empty # raw python value for item
		self.childModel = self.getNewChildModel() # model for all child items
		# try to keep childModel persistent as object
		#print("init", type(self), value)
		#self.setValue(value)

	_reprDepth = 0

	def childSuperItems(self)->list[SuperItem]:
		return list(iterAllItems(model=self.childModel))


	# def __repr__(self):
	# 	outer = "\t" * self._reprDepth + f"""<{self.__class__.__name__}>({self.superItemValue}"""
	# 	SuperItem._reprDepth += 1
	# 	childRepr = [repr(i) for i in self.childSuperItems()]
	# 	SuperItem._reprDepth -= 1
	# 	#end = ")"
	# 	return "\n".join([outer] + childRepr)


	@classmethod
	def forValue(cls, value)->SuperItem:
		"""NO OVERRIDE
		creates item for value, and sets value on item.
		Outer and system interface, for getting plugin
		items for any types
		"""
		itemCls = cls.getPlugin(value)
		assert itemCls is not None, f"no item class for {type(value)},\n"  +\
		                            pprint.pformat(cls.pluginRegister.pluginMap)
		item = itemCls._getNewInstanceForValue(value)
		item.setValue(value)
		return item

	@classmethod
	def _getNewInstanceForValue(cls, value)->SuperItem:
		"""OVERRIDE if needed
		delegate to specific item classes, to do
		any specific setup around a new instance -
		DO NOT set value on item, that's handled by outer .forValue() method
		"""
		return cls()

	# def childSuperItems(self)->list [SuperItem]:
	# 	return [self.childModel.item(i, 0) for i in range(self.childModel.rowCount())]

	def clearValue(self):
		self.childModel.clear()

	def getNewChildModel(self):
		"""create new child model object and return it -
		don't set items on it yet"""
		model = self.modelCls()
		model.parentSuperItem = self
		return model

	def setValue(self, value):
		self.clearValue()
		self.superItemValue = value
		self.makeChildItems()

	def makeChildItems(self):
		"""run after setting value, create items for
		child entries and add them to childModel
		"""
		pass

	def getResultValue(self):
		""" OVERRIDE
		return new value from content of child widgets -
		"""
		raise NotImplementedError

	def getNewView(self, parentQObject=None)->viewCls:
		"""create a new view class, set it on this item's model,
		return it
		"""
		#log("getNewView", self, self.viewCls, self.childModel)
		if not self.viewCls:
			return None
		view = self.viewCls(parent=parentQObject)
		view.parentSuperItem = self
		view.setModel(self.childModel)
		return view


