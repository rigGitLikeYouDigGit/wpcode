
from __future__ import annotations

from PySide2 import QtCore, QtWidgets, QtGui

from wplib import log

from wpdex import WpDex, DictDex, WpDexProxy
from wpdex.ui.atomic import AtomicWidget
from wpdex.ui.base import WpDexView, DexViewExpandButton



class _DictDexModel(QtGui.QStandardItemModel):
	pass


#class SeqDexView(AtomicWidget, QtWidgets.QTreeView):
class DictDexView(QtWidgets.QTreeView,# AtomicWidget
                 WpDexView
                 ):
	"""view for dict
	"""
	forTypes = (DictDex,)

	def __init__(self, value, parent=None):
		QtWidgets.QTreeView.__init__(self, parent)
		WpDexView.__init__(self, value)
		#log("seq init")
		a = 1

		self.header().setDefaultSectionSize(2)
		#self.header().setMinimumSectionSize(-1) # sets to font metrics, still buffer around it
		self.header().setMinimumSectionSize(15)
		self.header().setSectionResizeMode(
			self.header().ResizeToContents
		)
		self.setColumnWidth(0, 2)
		self.setIndentation(0)

		self.setSizeAdjustPolicy(
			QtWidgets.QAbstractItemView.AdjustToContents)
		self.setHeaderHidden(True)

		self.setVerticalScrollMode(self.ScrollMode.ScrollPerPixel)
		self.setHorizontalScrollMode(self.ScrollMode.ScrollPerPixel)

		self.postInit()
		self.setContentsMargins(0, 0, 0, 0)
		self.setViewportMargins(0, 0, 0, 0)

		self.setUniformRowHeights(False)


	def modelCls(self):
		return _DictDexModel

	def _modelIndexForKey(self, key:WpDex.keyT)->QtCore.QModelIndex:
		if "key:" in str(key):
			index = tuple(self.dex().branchMap().keys()).index(key) // 2
			return self.model().index(index, 1)
		index = tuple(self.dex().branchMap().keys()).index(key)
		return self.model().index(index, 2)
	def buildChildWidgets(self):
		"""populate childWidgets map with widgets
		for all dex children"""
		#log("buildChildWidgets")
		self.setModel(self.modelCls()(parent=self))
		items = tuple(self.dex().branchMap().items())
		keyTies = [i for i in items if "key:" in str(i[0])]
		valueTies = [i for i in items if "key:" not in str(i[0])]
		for i, ((keyKey, keyDex),
		     (valueKey, valueDex)) in enumerate(zip(keyTies, valueTies)):
			keyWidgetType = AtomicWidget.adaptorForObject(keyDex)
			assert keyWidgetType
			valueWidgetType = AtomicWidget.adaptorForObject(valueDex)
			assert valueWidgetType
			keyWidget = keyWidgetType(value=keyDex, parent=self)
			valueWidget = valueWidgetType(value=valueDex, parent=self)

			self.model().appendRow(
				[QtGui.QStandardItem(),
				 QtGui.QStandardItem(str(keyDex.obj)),
				 QtGui.QStandardItem(str(valueDex.obj))]
			)
			keyIndex = self._modelIndexForKey(keyKey)
			self.setIndexWidget(keyIndex, keyWidget)
			self._setChildAtomicWidget(keyKey, keyWidget)

			valueIndex = self._modelIndexForKey(valueKey)
			self.setIndexWidget(valueIndex, valueWidget)
			self._setChildAtomicWidget(valueKey, valueWidget)

		# rootItem : QtGui.QStandardItem = self.model().itemFromIndex(self.model().index(0, 0))
		# rootItem.setText("[")
		topLeftIndex = self.model().index(0, 0)
		label = DexViewExpandButton("{", dex=self.dex(), parent=self)
		label.expanded.connect(self._setValuesVisible)
		#label.clicked.connect(self._toggleValuesVisible)

		self.setIndexWidget(topLeftIndex, label)
		self.syncLayout()

	def _setValuesVisible(self, state=True):
		self.setColumnHidden(1,
		                     not state
		                     )
		self.setColumnHidden(2,
		                     not state
		                     )
		self.resizeColumnToContents(0)
		self.resizeColumnToContents(1)
		self.update()
		self.updateGeometry()
		self.syncLayout()
		self.parent().updateGeometry()
		if isinstance(self.parent(), WpDexView):
			self.parent().syncLayout()
			self.parent().updateGeometries()
			self.parent().syncLayout()

if __name__ == '__main__':


	from wpdex.ui.base import WpDexWindow
	d = {"strkey" : "value",
	     4 : 5}
	p = WpDexProxy(d)
	dex = p.dex()
	log(dex, dex.branchMap())


	ref = p.ref()
	log("ref", ref, "ref val", ref.rx.value)

	app = QtWidgets.QApplication()
	w = WpDexWindow(parent=None,
	                value=ref)
	w.show()
	app.exec_()


