


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

if T.TYPE_CHECKING:
	from wptree import Tree, TreeInterface
	from wptree.ui import TreeSuperItem


ITEM_PADDING = 10

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


	it SEEMS that nested items inherit their sizehint directly from their root


	"""

	# def __init__(self, parent=None):
	# # 	super(SuperDelegate, self).__init__(parent)
	#
	def _sizeHintForIndex(self, option:QtWidgets.QStyleOptionViewItem, index:QtCore.QModelIndex) -> PySide2.QtCore.QSize:
		"""return size for single index"""
		#log("size hint for index", index, index.model().itemFromIndex(index))
		if self.parent().indexWidget(index):
			#log("index widget size", self.parent().indexWidget(index), self.parent().indexWidget(index).sizeHint())
			return self.parent().indexWidget(index).sizeHint()
			return self.parent().indexWidget(index).sizeHint().grownBy(
				QtCore.QMargins(ITEM_PADDING, ITEM_PADDING, ITEM_PADDING, ITEM_PADDING)
			)
		item = index.model().itemFromIndex(index)
		baseSize = super(SuperDelegate, self).sizeHint(option, index)
		return baseSize
		pass
	def sizeHint(self, option:PySide2.QtWidgets.QStyleOptionViewItem, index:PySide2.QtCore.QModelIndex) -> PySide2.QtCore.QSize:
		"""return size hint for index - if complex, delegate to nested widget

		widget sizes are being reported properly - seems like setting
		heights on branch items just doesn't do anything if the
		root isn't set too?
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
		#height = 60
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
			#log("height", height, indexSize.height(), max(height, indexSize.height()))
			#log("item", item, indexSize, indexSize.height(), "for", childIndex.model().itemFromIndex(childIndex))
			height = max(height, indexSize.height())

			#width += indexSize.width()

		#log("endSizeHint for", item, QtCore.QSize(width, height), "rowItems", rowItems)
		return QtCore.QSize(width, height)


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
		elif role == QtCore.Qt.DisplayRole:
			if orientation == QtCore.Qt.Horizontal:
				if section == 0:
					return ""
				# elif section == 1:
				# 	return "value"
		return super(SuperModel, self).headerData(section, orientation, role)
	#


class TypeLabel(QtWidgets.QLabel):
	"""label for type, occupies first column of header -
	clicking will expand or collapse full view"""

	clicked = QtCore.Signal(dict)

	def __init__(self, *args, **kwargs):
		super(TypeLabel, self).__init__(*args, **kwargs)
		self.setMargin(2)
		self.setContentsMargins(0, 0, 0, 0)
		#self.setIndent(0)
		#self.setFrameStyle(QtWidgets.QFrame.Raised)
		#self.setStyleSheet("QLabel { padding: 0px; margin: 0px; border-radius: 5px}")
		# self.setStyleSheet("")
		# self.setAutoFillBackground(True)
		# palette = QtGui.QPalette()
		# palette.setColor(QtGui.QPalette.Background, QtCore.Qt.gray)
		#
		# self.setPalette(palette)

	def paintEvent(self, arg__1:PySide2.QtGui.QPaintEvent) -> None:
		"""draw background"""
		painter = QtGui.QPainter(self)
		painter.setPen(QtCore.Qt.NoPen)
		painter.setBrush(QtGui.QBrush(QtCore.Qt.gray))
		painter.drawRoundedRect(self.rect(), 2, 2)

		painter.setPen(QtGui.QPen(QtCore.Qt.white))
		painter.setFont(self.font())
		painter.drawText(self.rect(), QtCore.Qt.AlignCenter, self.text())
		#super(TypeLabel, self).paintEvent(arg__1)

		pass

	def mousePressEvent(self, ev:PySide2.QtGui.QMouseEvent) -> None:
		"""emit signal with event"""
		#log("clicked", ev)
		self.clicked.emit({"event": ev})


base = object
if T.TYPE_CHECKING:
	base = WPTableView
# else:
# 	base = object



class SuperViewBase(
	base
                    ):
	"""base class for view widgets into nested objects -
	each list, dict or tree is has its own view, with views
	of child items being index widgets in this view

	first column of header should be a label with the class name -
	also functions to collapse view when not wanted
	"""

	parentSuperItem : SuperItem

	#sizeChanged = QtCore.Signal(QtCore.QSize)

	viewExpanded = QtCore.Signal(bool)

	def __init__(self:QtWidgets.QAbstractItemView, *args, **kwargs):
		#super(SuperViewBase, self).__init__(*args, **kwargs)
		self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
		self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
		self.setAlternatingRowColors(True)
		#self.setViewportMargins(0, 0, 0, 0)
		#self.setViewportMargins(ITEM_PADDING, ITEM_PADDING, ITEM_PADDING, ITEM_PADDING)
		#self.setContentsMargins(0, 0, 0, 0)

		self.setContentsMargins(ITEM_PADDING, ITEM_PADDING, ITEM_PADDING, ITEM_PADDING)
		#self.setFrameShape(QtWidgets.QFrame.NoFrame)
		#self.setFrameShape(QtWidgets.QFrame.StyledPanel)

		self.setItemDelegate(SuperDelegate(self))

		self.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
		self.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
		self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
		self.setFrameStyle(QtWidgets.QFrame.NoFrame)

		try:
			self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
		except AttributeError:
			pass
		try:
			self.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
		except AttributeError:
			pass

		self._expanded = True

		self.typeLabel = self.makeTypeLabel()
		self.typeLabel.clicked.connect(self._onTypeLabelClicked)


		#self.typeLabel.setText(type(self.parentSuperItemValue).__name__)


		#self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
		#self.setWidgetResizable(True)



	def _getTopHeader(self):
		"""tableView defines horizontalHeader and verticalHeader -
		treeView only defines header()"""
		try:
			return self.horizontalHeader()
		except AttributeError:
			return self.header()

	def makeTypeLabel(self):
		label = TypeLabel(parent=self._getTopHeader())
		font = label.font()
		font.setPointSize(7)
		label.setFont(font)
		#label.setFont(QtGui.QFont("Courier", 2))
		#label.setText(type(self.parentSuperItemValue).__name__)
		return label


	def setExpanded(self, state:bool):
		"""expand or collapse view"""
		#log("setExpanded", self.isExpanded(), state)
		self._expanded = state
		if self._expanded:
			self._expand()
		else:
			self._collapse()
		self.viewExpanded.emit({"state": state})
		# self.update()
		# # self.regenWidgets()
		# self.syncChildWidgetSizes()

	def _expand(self:QtWidgets.QWidget):
		#self.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
		# self.setMinimumSize(...)
		# self.setMaximumSize(...)
		self.setMaximumSize(QtCore.QSize(16777215, 16777215))
		self.setMinimumSize(self.sizeHint())

	def _collapse(self):
		self.setMinimumSize(self.typeLabel.sizeHint())
		self.setMaximumSize(self.typeLabel.sizeHint())

	def isExpanded(self):
		return self._expanded

	def toggleExpanded(self):
		self.setExpanded(not self.isExpanded())

	def _onTypeLabelClicked(self, *args, **kwargs):
		self.toggleExpanded()

	def _onChildExpanded(self, *args, **kwargs):
		"""child view expanded - update size"""
		#log("child expanded", self)
		self.syncChildWidgetSizes()
		# try:
		# 	self.resizeRowsToContents()
		# except AttributeError:
		# 	pass
		self.doItemsLayout()

	def sizeHint(self) -> PySide2.QtCore.QSize:
		if self._expanded:
			return super(SuperViewBase, self).sizeHint()
		return self.typeLabel.sizeHint()

	if T.TYPE_CHECKING:
		def model(self)->SuperModel:
			pass


	def onItemChanged(self, item:QtGui.QStandardItem):
		"""item changed - update view"""
		self.regenWidgets()
		self.syncChildWidgetSizes()
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
		try:
			self.resizeRowsToContents()
		except AttributeError:
			pass
		self.updateGeometries()
		self.updateGeometry()

		self.updateTypeLabel()

	def updateTypeLabel(self):

		self.typeLabel.setText(type(self.parentSuperItem.
		                            superItemValue).__name__)
		self.typeLabel.move(2, 2)



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
				w.viewExpanded.connect(self._onChildExpanded)
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
				for index in self.model().indicesForWidgets()]

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

	def data(self, role:int=...) -> typing.Any:
		if role == QtCore.Qt.TextAlignmentRole:
			return QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop
		return super(SuperItem, self).data(role)


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


