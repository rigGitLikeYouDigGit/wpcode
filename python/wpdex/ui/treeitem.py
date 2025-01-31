
from __future__ import annotations
import typing as T
import pprint
import weakref

from PySide2 import QtCore, QtWidgets, QtGui

from wplib import log
from wptree import TreeInterface, TreeDex, Tree
from wpdex import WpDex, DictDex, WpDexProxy
from wpdex.ui.atomic import AtomicWidgetOld
from wpdex.ui.atomic.base import AtomicStandardItemModel, AtomStandardItem, AtomStyledItemDelegate, AtomicUiInterface, AtomicWindow, AtomicView


class TreeBranchItem(AtomStandardItem):
	forTypes = (TreeDex, )
	"""branch item is editable for the name 
	string of a tree, but affects tree structure
	if you drag or duplicate it
	
	"""

	def __init__(self, value=None,
	             dex:TreeDex=None):
		self.treeDex = dex
		AtomStandardItem.__init__(self, value)
		#self.__post_init__()

	def valueItem(self)->AtomStandardItem:
		if self.parent():
			return self.model().itemFromIndex(
				self.model().index(
					self.row(), 1, self.parent().index()
				)
			)
		return self.model().itemFromIndex(
			self.model().index(
				self.row(), 1,
			)
		)


class TreeDexModel(AtomicStandardItemModel):
	"""will probably need this for drag/drop
	at some point
	"""
	forTypes = (TreeDex, )

	if T.TYPE_CHECKING:
		def dex(self) -> TreeDex:...

	def __init__(self, value, parent=None):
		self.uidBranchItemMap : dict[str, TreeBranchItem] = weakref.WeakValueDictionary()
		AtomicStandardItemModel.__init__(self, value=value, parent=parent)

	def _modelIndexForKey(self, key:WpDex.pathT)->QtCore.QModelIndex:
		key = WpDex.toPath(key)
		if "key:" in str(key[0]):
			index = tuple(self.dex().branchMap().keys()).index(key) // 2
			return self.model().index(index, 1)
		index = tuple(self.dex().branchMap().keys()).index(key)
		return self.model().index(index, 2)

	# def _buildItemsForBranch(self, branch:Tree)->list[QtGui.QStandardItem]:
	# 	return [
	#
	# 	]

	def _buildItems(self):
		""" OVERRIDE
		create standardItems for each branch of
		this atom's dex
		for each one, check if it's a container - if so, make
		its own child view widget

		models built out in this function - VIEW has to INDEPENDENTLY
		construct and match up child view widgets for
		each entry that needs them?
		that might actually be the least complicated
		"""
		self.uidBranchItemMap.clear()
		self.clear()
		#self._clearChildWidgets()

		#TODO: possible we should give each WpDex a uid too

		#ERROR when we initialise on a tree not at root - fix
		for treeDex in self.dex().treeDexes(includeSelf=True):
			parentItem = self.invisibleRootItem()
			if treeDex.parent:
				if isinstance(treeDex.parent, TreeDex):
					parentItem = self.uidBranchItemMap.get(
						treeDex.parent.obj.uid
					) or parentItem # if None, we're initialised on a lower parent tree, use the root

			branchItem = TreeBranchItem(value=treeDex.branchMap()["@N"],
			                            dex=treeDex)

			valueItemType = AtomStandardItem.adaptorForObject(treeDex.branchMap()["@V"])
			valueItem = valueItemType(value=treeDex.branchMap()["@V"])
			self.uidBranchItemMap[treeDex.obj.uid] = branchItem
			parentItem.appendRow([
				branchItem, valueItem
			])


class TreeDexView(#QtWidgets.QTreeView,# AtomicWidget
                 AtomicView
                 ):
	"""view for tree -
	I believe this will be my 5th attempt at making a ui for this thing

	not much interaction with wpdex left for view, manage that mainly in
	model and item
	"""
	if T.TYPE_CHECKING:
		def dex(self)->TreeDex:...
	forTypes = (TreeDex,)


if __name__ == '__main__':


	#from wpdex.ui.base import WpDexWindow
	d = Tree("root", value=3)
	d["branchA"] = "branchAValue"
	d["branchA", "leaf"] = 33
	d["branchB", "leaf"] = ["a", Tree("valueTreeName", value=44,
	                                  branches=([Tree("valueBranch",
	                                                  branches=[Tree("value leaf", value=":)")])]))]
	d["branchB"] = {"dict key" : 22}
	p = WpDexProxy(d)
	dex = p.dex()

	#assert d("branchA", "leaf").root is d
	assert dex.root is dex
	for i in dex.treeDexes():
		assert i.root is dex
	#log(dex, dex.branchMap())
	#pprint.pp(dex.branchMap())
	#
	#
	ref = p.ref()
	# log("ref", ref, "ref val", ref.rx.value)
	#
	app = QtWidgets.QApplication()
	w = AtomicWindow(parent=None,
	                 value=ref)
	w.show()
	app.exec_()


