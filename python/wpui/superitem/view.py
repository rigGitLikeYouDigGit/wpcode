
from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from wplib.constant import LITERAL_TYPES, SEQ_TYPES, MAP_TYPES
from wplib.sentinel import Sentinel
from wplib.object import PluginBase

from wpui.superitem.delegate import SuperDelegate

if T.TYPE_CHECKING:
	from wpui.superitem.model import SuperModel
	from wpui.superitem.combined import SuperItemBase


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

	if T.TYPE_CHECKING:
		def model(self)->SuperModel:
			pass

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

			delegate.sizeHintChanged.emit(index)
			view.syncSize()

		self.syncSize()

		pass

	def parentSuperItem(self)->SuperItemBase:
		return self.model().parentSuperItem

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

	def resizeEvent(self, event):
		#self.sizeChanged.emit(self.size())
		for i in self.childViews():
			i.resizeEvent(event)
		self.syncSize()

		result = super(SuperViewBase, self).resizeEvent(event)
		return result

