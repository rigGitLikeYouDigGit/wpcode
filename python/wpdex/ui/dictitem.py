
from __future__ import annotations

from PySide2 import QtCore, QtWidgets, QtGui

from wplib import log
from wplib.inheritance import MetaResolver
from wpdex import WpDex, SeqDex, WpDexProxy, DictDex
from wpdex.ui.atomic import AtomicWidgetOld, AtomicView, AtomicStandardItemModel, AtomStandardItem

from wpdex.ui.base import WpDexView#, DexViewExpandButton



class DictDexModel(AtomicStandardItemModel):
	forTypes = (DictDex, )
	def _buildItems(self):
		self.clear()
		items = tuple(self.dex().branchMap().items())
		keyTies = [i for i in items if "key:" in str(i[0])]
		valueTies = [i for i in items if "key:" not in str(i[0])]
		for i, ((keyKey, keyDex),
		     (valueKey, valueDex)) in enumerate(zip(keyTies, valueTies)):
			keyItemType = AtomStandardItem.adaptorForObject(keyDex)
			assert keyItemType
			valueItemType = AtomStandardItem.adaptorForObject(valueDex)
			assert valueItemType
			keyItem = keyItemType(value=keyDex)
			valueItem = valueItemType(value=valueDex)
			self.appendRow([keyItem, valueItem])


class DictDexView(MetaResolver, QtWidgets.QTreeView,# AtomicWidget
                 WpDexView
                 ):
	"""view for dict
	"""
	forTypes = (DictDex,)

	def __init__(self, value, parent=None):
		QtWidgets.QTreeView.__init__(self, parent)
		WpDexView.__init__(self, value)

		self.postInit()


	def _modelIndexForKey(self, key:WpDex.keyT)->QtCore.QModelIndex:
		if "key:" in str(key):
			index = tuple(self.dex().branchMap().keys()).index(key) // 2
			return self.model().index(index, 1)
		index = tuple(self.dex().branchMap().keys()).index(key)
		return self.model().index(index, 2)


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


