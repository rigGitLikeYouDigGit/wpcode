
from __future__ import annotations
import typing as T
import pprint
import weakref

from PySide2 import QtCore, QtWidgets, QtGui

from wplib import log
from wptree import TreeInterface, TreeDex, Tree
from wpdex import WpDex, DictDex, WpDexProxy
from wpdex.ui.atomic import AtomicWidget
from wpdex.ui.base import WpDexView, DexViewExpandButton



class _TreeDexModel(QtGui.QStandardItemModel):
	"""will probably need this for drag/drop
	at some point"""
	pass

class _TreeBranchItem(QtGui.QStandardItem):

	def __init__(self, dex:TreeDex, text:str=""):
		#self.branch = branch
		self.dex = dex
		QtGui.QStandardItem.__init__(self, text)
	def branch(self)->Tree:
		return self.dex.obj

	pass

class TreeDexView(QtWidgets.QTreeView,# AtomicWidget
                 WpDexView
                 ):
	"""view for tree -
	I believe this will be my 5th attempt at making a ui for this thing

	difficulty here is to have only one view for a whole tree, but
	a separate dex object for each branch
	separate items for values? could also try drawing them nested in with
	their branch
	"""
	if T.TYPE_CHECKING:
		def dex(self)->TreeDex:...
	forTypes = (TreeDex,)

	def __init__(self, value, parent=None):
		QtWidgets.QTreeView.__init__(self, parent)
		WpDexView.__init__(self, value)
		#log("seq init")
		a = 1

		self.uidBranchItemMap : dict[str, _TreeBranchItem] = weakref.WeakValueDictionary()
		self.postInit()



	def modelCls(self):
		return _TreeDexModel

	def _modelIndexForKey(self, key:WpDex.keyT)->QtCore.QModelIndex:
		if "key:" in str(key):
			index = tuple(self.dex().branchMap().keys()).index(key) // 2
			return self.model().index(index, 1)
		index = tuple(self.dex().branchMap().keys()).index(key)
		return self.model().index(index, 2)

	# def _buildItemsForBranch(self, branch:Tree)->list[QtGui.QStandardItem]:
	# 	return [
	#
	# 	]

	def buildChildWidgets(self):
		"""populate childWidgets map with widgets
		for all dex children
		lots of duplication here, logic is not elegant"""
		self.uidBranchItemMap.clear()
		self._clearChildWidgets()

		parentItem = _TreeBranchItem(
			#branch=self.dex().obj,
			dex=self.dex(),
			text=self.dex().obj.name
		)
		self.uidBranchItemMap[self.dex().obj.uid] = parentItem
		self.model().appendRow([
			QtGui.QStandardItem(),
			parentItem,
			QtGui.QStandardItem(#str(self.dex().obj.value)
				"tree_value"
			                    )
		])
		nameIndex = self.model().index(0, 1)
		w = self._buildChildWidget(nameIndex, self.dex().branchMap()["@N"])
		valueIndex = self.model().index(0, 2)
		w = self._buildChildWidget(valueIndex, self.dex().branchMap()["@V"])
		#self.setExpanded(nameIndex)
		#TODO: possible we should give each WpDex a uid too
		for treeDex in self.dex().treeDexes(includeSelf=False):
			parentItem = self.uidBranchItemMap[
				treeDex.parent.obj.uid
			]
			branchItem = _TreeBranchItem(dex=treeDex)
			parentItem.appendRow([
				branchItem, QtGui.QStandardItem("tree_value")
			])
			nameIndex = self.model().index(parentItem.rowCount()-1, 0, parentItem.index())
			self.uidBranchItemMap[treeDex.obj.uid] = branchItem
			valueIndex = self.model().index(parentItem.rowCount()-1, 1, parentItem.index())
			# w = self._buildChildWidget(nameIndex, treeDex.branchMap()["@N"])
			# w = self._buildChildWidget(valueIndex, treeDex.branchMap()["@V"])
			self.setExpanded(parentItem.index(), True)


		topLeftIndex = self.model().index(0, 0)
		label = DexViewExpandButton("t", dex=self.dex(), parent=self)
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
	d = Tree("root", value=3)
	d["branchA"] = "branchAValue"
	d["branchA", "leaf"] = 33
	# d["branchB"] = [3, 4, 5]
	p = WpDexProxy(d)
	dex = p.dex()
	#log(dex, dex.branchMap())
	#pprint.pp(dex.branchMap())
	#
	#
	ref = p.ref()
	# log("ref", ref, "ref val", ref.rx.value)
	#
	app = QtWidgets.QApplication()
	w = WpDexWindow(parent=None,
	                value=ref)
	w.show()
	app.exec_()


