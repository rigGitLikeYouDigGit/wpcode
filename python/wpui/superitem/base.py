


from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from wplib.constant import LITERAL_TYPES, SEQ_TYPES, MAP_TYPES
from wplib.sentinel import Sentinel
from wplib.object import PluginRegister, PluginBase

from wpui.model import iterAllItems

from wpui.widget import WPTableView

"""what if whole thing in one"""


class SuperDelegate(QtWidgets.QStyledItemDelegate):
	"""delegate for superitem"""

	# def __init__(self, parent=None):
	# 	super(SuperDelegate, self).__init__(parent)

	def sizeHint(self, option:PySide2.QtWidgets.QStyleOptionViewItem, index:PySide2.QtCore.QModelIndex) -> PySide2.QtCore.QSize:
		"""return size hint for index - if complex, delegate to nested widget
		"""
		#print("size hint", index)
		if self.parent().indexWidget(index):
			return self.parent().indexWidget(index).sizeHint()
			item : SuperItem = self.parent().model().itemFromIndex(index)
			#print("size hint", item, item.childWidget.sizeHint())
			if item.childWidget:
				return item.childWidget.sizeHint()

		return super(SuperDelegate, self).sizeHint(option, index)


class SuperModel(QtGui.QStandardItemModel):
	"""not the photographic kind, this is way better"""

	def __init__(self):
		super(SuperModel, self).__init__()
		self.parentSuperItem : SuperItem = None

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
		self.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)

	if T.TYPE_CHECKING:
		def model(self)->SuperModel:
			pass

	def sizeHint(self):
		"""combine size hints of all rows and columns"""

		x = self.size().width()
		y = 0

		for i in range(self.model().rowCount()):
			y += self.sizeHintForRow(i) + 2

		try:
			y += self.horizontalHeader().height()
		except:
			pass
		#y += 20
		total = QtCore.QSize(x, y)
		return total


	def setModel(self, model: SuperModel):
		"""run over model to set view widgets on delegates"""
		delegate = SuperDelegate(self)
		self.setItemDelegate(delegate)
		super(SuperViewBase, self).setModel(model)

		for item in iterAllItems(model=model):
			if not isinstance(item, SuperItem):
				continue
			if item.childModel is not None:
				self.setIndexWidget(item.index(), item.getNewView())
			if item.delegateCls is not None:
				delegate = item.delegateCls(self)

				self.setItemDelegateForColumn(item.column(), delegate)

			#delegate.sizeHintChanged.emit(item.index())
			#view.syncSize()

		self.syncSize()

		pass


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

	def __init__(self, value):
		super(SuperItem, self).__init__()
		self.value = Sentinel.Empty # raw python value for item
		self.childModel = self.getNewChildModel() # model for all child items
		# try to keep childModel persistent as object
		#print("init", type(self), value)
		self.setValue(value)

	_reprDepth = 0
	def __repr__(self):
		outer = "\t" * self._reprDepth + f"""<{self.__class__.__name__}>({self.value}"""
		SuperItem._reprDepth += 1
		childRepr = [repr(i) for i in self.childSuperItems()]
		SuperItem._reprDepth -= 1
		#end = ")"
		return "\n".join([outer] + childRepr)


	@classmethod
	def forValue(cls, value)->SuperItem:
		itemCls = cls.getPlugin(value)
		return itemCls(value)

	# def childSuperItems(self)->list[SuperItem]:
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
		self.value = value
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

	def getNewView(self)->viewCls:
		"""create a new view class, set it on this item's model,
		return it
		"""
		print("getNewView", self, self.viewCls, self.childModel)
		view = self.viewCls()
		view.parentSuperItem = self
		view.setModel(self.childModel)
		return view


