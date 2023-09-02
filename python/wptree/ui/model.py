
from __future__ import annotations
import typing as T
import weakref
from itertools import product

from PySide2 import QtGui, QtCore, QtWidgets

from wplib.object import UidElement

from wptree.main import Tree
from wptree.ui.constant import addressRole, relAddressRole, treeObjRole
from wptree.ui.item import TreeValueItem, TreeBranchItem


"""test for something slightly more advanced - 
each model is initialised on a tree root, and any views of any branches of that
tree attach to that same model."""

class TreeModel(QtGui.QStandardItemModel, UidElement):


	beforeItemChanged = QtCore.Signal(QtGui.QStandardItem)
	afterItemChanged = QtCore.Signal(QtGui.QStandardItem)

	# create new instance map for all tree models
	indexInstanceMap = {}

	def __init__(self, tree=None, parent=None,
	             # branchItemCls=TreeBranchItem,
	             # branchValueCls=TreeValueItem,
	             ):
		QtGui.QStandardItemModel.__init__(self, parent)
		UidElement.__init__(self)

		self.treeRef : weakref.ReferenceType[Tree] = None
		self.setHorizontalHeaderLabels(["branch", "value"])

		if tree:
			self.setTree(tree)


	@property
	def tree(self)->Tree:
		if self.treeRef is None:
			return None
		assert self.treeRef() is not None, "tree has been deleted"
		return self.treeRef()

	def getBranchItemCls(self, forBranch:Tree):
		return TreeBranchItem

	def getBranchValueCls(self, forBranch:Tree):
		return TreeValueItem

	def setTree(self, tree):
		"""we enforce initialising a model on a tree root.
		Update this"""
		self.clear()
		tree = tree.root
		self.treeRef = weakref.ref(tree)
		self.setElementId(tree.uid)
		items = self.getBranchItemCls(tree).itemsForBranch(
			tree)
		self.appendRow(items)

	def rootItem(self)->TreeBranchItem:
		return self.item(0)

	@classmethod
	def modelForTree(cls, tree:Tree, parent:QtCore.QObject=None):
		"""return the model for a tree, or create one if it doesn't exist"""
		#print("model for tree", tree.root.uid)
		if tree.root.uid in cls.indexInstanceMap:
			print("returning existing model", cls.indexInstanceMap[tree.root.uid])
			return cls.indexInstanceMap[tree.root.uid]
		#print("creating new model")
		model = cls(tree=tree, parent=parent)

		return model

	# def beforeBranchSync(self, item:TreeBranchItem):
	# 	"""fire this method manually before any automaed tree changes
	# 	mixing up order of signals like this is some sweater spaghetti
	# 	but it's fine
	# 	"""
	# 	self.beforeItemChanged.emit(item)
	# def afterBranchSync(self, item:TreeBranchItem):
	# 	self.afterItemChanged.emit(item)


	# drag and drop support
	def supportedDropActions(self):
		return QtCore.Qt.MoveAction

	def mimeTypes(self):
		""" use built in abstractTree serialisation to reduce
		entries to plain text, then regenerate them after """
		types = ["text/plain"]
		return types

	def mimeData(self, indices):
		# indices = indices[0::2]
		# filter only branchItems, not values
		indices = filter(lambda x: isinstance(
			self.itemFromIndex(x), TreeBranchItem), indices)
		infos = []
		for i in indices:
			branchItem = self.itemFromIndex(i)
			branch = branchItem.tree
			info = branch.serialise(includeAddress=True)
			infos.append(info)
		text = str(infos)
		mime = QtCore.QMimeData()
		mime.setText(text)
		return mime

	def recursiveRows(self, topRow:QtCore.QModelIndex, includeTop=True, bottomUp=False)->list[QtCore.QModelIndex]:
		"""return list of child rows below the first"""
		rows = [topRow] if includeTop else []
		for row in range(topRow.model().rowCount(topRow)):
			rows.extend(self.recursiveRows(topRow.child(row, 0), includeTop=True, bottomUp=bottomUp))
		if bottomUp:
			rows.reverse()
		return rows


	def dropMimeData(self, data, action, row, column, parentIndex):
		""" used for dropping and pasting """
		if action == QtCore.Qt.IgnoreAction:
			return True
		if not data.hasText():
			return False
		mimeText = data.text()
		infos = eval(mimeText)
		if not isinstance(infos, list):
			infos = [infos]

		for info in infos:
			tree = Tree.fromDict(info)

			# remove original entries
			if action == QtCore.Qt.MoveAction and "?ADDR" in info:
				found = self.tree.getBranch(info["?ADDR"])
				if found:
					found.remove()

			parentItem = self.itemFromIndex(parentIndex)
			if not parentItem:
				#parentItem = self.invisibleRootItem()
				#parentTree = self.tree.root
				parentTree = self.tree
			else:
				parentTree = parentItem.tree
			parentTree.addChild(tree)

		#self.sync()
		return True

	def branchFromIndex(self, index):
		""" returns tree object associated with qModelIndex """
		return self.itemFromIndex(index).tree

	def connectedIndices(self, index):
		""" return previous, next, upper and lower indices
		or None if not present
		only considers rows for now """
		result = {}
		nRows = self.rowCount(index.parent())
		nextIdx = index.sibling((index.row() + 1) % nRows, 0)
		result["next"] = nextIdx if nextIdx.isValid() else None
		prevIdx = index.sibling((index.row() - 1) % nRows, 0)
		result["prev"] = prevIdx if prevIdx.isValid() else None
		result["parent"] = index.parent() \
			if not index.parent() == QtCore.QModelIndex() else None
		return result

	@staticmethod
	def allChildren(index:QtCore.QModelIndex):
		result = [index]
		for row, column in product(
				range(index.model().rowCount(index)),
				range(index.model().columnCount(index)),
				      ):
			result.extend(TreeModel.allChildren(index.child(row, column)))
		return result

	@staticmethod
	def rowFromIndex(index):
		""" return the row index for either row or value index """
		return index.parent().child(index.row(), 0)

	def allRows(self, _parent=None)->T.List[QtCore.QModelIndex]:
		""" return flat list of all row indices """

		if _parent is None: _parent = QtCore.QModelIndex()
		rows = []

		for i in range(self.rowCount(_parent)):
			index = self.index(i, 0, _parent)
			rows.append(index)
			rows.extend(self.allRows(index))

		return rows

	def rowFromTree(self, tree:Tree)->QtCore.QModelIndex:
		""" returns index corresponding to tree
		inelegant search for now """
		#print("row from tree {}".format(tree))
		for i in self.allRows():
			#print("item branch", self.treeFromRow(i))
			#print("found {}".format(self.treeFromRow(i)))
			if self.treeFromRow(i) is tree:
				#print("found match", tree, i)
				return i

	def treeFromRow(self, row:QtCore.QModelIndex)->Tree:
		""":rtype Tree """
		# return self.tree.getBranch(self.data(row, objRole))
		# print("tree", self.tree.displayStr())
		# print("from row", self.data(row, relAddressRole))
		return self.data(row, treeObjRole)
		#return self.tree.getBranch(self.data(row, relAddressRole))

	def duplicateRow(self, row:QtCore.QModelIndex)->Tree:
		""" copies tree, increments its name, adds it as sibling
		:param row : QModelIndex for row """
		# print("")
		# print("model duplicate row")

		address = self.data(row, relAddressRole)

		tree = self.tree(address)
		treeParent : Tree = tree.parent
		newTree = tree.copy()

		# assign non-duplicate name for new tree
		newTree.name = treeParent.getUniqueBranchName(newTree.name)
		treeParent.addChild(newTree)
		return newTree



	def shiftRow(self, row:QtCore.QModelIndex, up=True):
		""" shifts row within its siblings up or down """


		tree = self.treeFromRow(row)
		#print("model shiftRow", tree)

		parent = tree.parent
		if not parent: # shrug
			return
		startIndex = tree.index()
		#print("start index", startIndex)
		newIndex = max(0, min(len(parent.branches), startIndex + (-1 if up else 1)))
		tree.setIndex(newIndex)
		#print("new index", newIndex, tree.index())


	def deleteRow(self, row):
		""" removes tree branch, then removes item """
		#tree = self.tree(self.data(row, objRole))
		#print("")
		toRemove : Tree = self.tree(self.data(row, relAddressRole))
		parent = toRemove.parent
		#print("removing", toRemove, toRemove.frameContextEnabled())
		toRemove.remove()
		#print("after remove", parent.branches)



	def unParentRow(self, row):
		""" parent row to its parent's parent """
		#branch = self.tree(self.data(row, objRole))
		print("")
		branch = self.tree(self.data(row, relAddressRole))
		parent = branch.parent
		if parent:
			grandparent = parent.parent
			if grandparent:
				print("unparent row", branch)
				branch.remove()
				#grandparent.addChild(branch)
				grandparent.addChild(branch, index=parent.index() + 1)
				print("found new row", self.rowFromTree(branch))

	def parentRows(self, rows:list[QtCore.QModelIndex], target:QtCore.QModelIndex):
		""" parent all selected rows to last select target """
		# parent = self.tree(self.data(target, objRole))
		parent = self.tree(self.data(target, relAddressRole))
		for i in rows:
			# branch = self.tree(self.data(i, objRole))
			branch = self.tree(self.data(i, relAddressRole))
			# print("parent", parent)
			# print("branch parent", branch.parent)
			if branch.parent is parent:
				#print("branch parent is parent")
				continue
			branch.remove()
			print("add child", branch)
			parent.addChild(branch)

