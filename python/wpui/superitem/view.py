
from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from wplib.constant import LITERAL_TYPES, SEQ_TYPES, MAP_TYPES
from wplib.sentinel import Sentinel

from wpui.superitem.delegate import SuperDelegate

if T.TYPE_CHECKING:
	from wpui.superitem.model import SuperModel
	from wpui.superitem.item import SuperItem


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
		#self.setFrameShape(QtWidgets.QFrame.NoFrame)
		self.setFrameShape(QtWidgets.QFrame.StyledPanel)
		self.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)

		#self.setSectionResizeMode(QtWidgets.QAbstractScrollArea.ResizeToContents)

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

	def setModel(self, model: SuperModel):
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
			y += self.sizeHintForRow(i) + 2

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



