from __future__ import annotations
import typing as T
import os, pathlib, weakref

from PySide2 import QtGui, QtCore

from wplib.constant import LITERAL_TYPES

from wptree.main import Tree
from wptree.ui.constant import addressRole, relAddressRole, childBoundsRole, treeObjRole, rowHeight


if T.TYPE_CHECKING:
	from tree.ui.model import TreeModel

"""for these ui items, we take an immediate approach of syncing and items directly
whenever tree changes
it's just so much simpler
 """



class TreeValueItem(QtGui.QStandardItem):
	""""""

	def __init__(self, tree:Tree):
		self.treeRef = weakref.ref(tree)
		super(TreeValueItem, self).__init__(
			self.processValueForDisplay(self.tree.value)
		)

	def getTree(self)->Tree:
		assert self.treeRef() is not None, "tree is dead"
		return self.treeRef()

	@property
	def tree(self):
		return self.getTree()

	def trueValue(self):
		return self.getTree().value

	def branchItem(self)->TreeBranchItem:
		return self.parent().child(self.row(), 0)

	def onTreeStateChanged(self, branch:Tree):
		if branch is not self.getTree():
			return
		self.setData(branch.value, role=2)


	def processValueForDisplay(self, value):
		""" strip inner quotes from container values -
		this is the raw data, separate from any fancy rendering later
		"""
		if value is None:
			return ""
		return str(value)

	def processValueFromDisplay(self, value):
		"""return displayed or entered text value
		to real type -
		don't change the type of the tree value if ambiguous
		"""
		if not value:
			if self.trueValue() is None:
				return None
		# if isinstance(value, LITERAL_TYPES):
		# 	return value
		return str(value)

	def setData(self, value, role=2):
		""""""
		if role == 2: # user role
			self.tree.value = value
			valueObj = self.processValueForDisplay(value)

			return super(TreeValueItem, self).setData(valueObj, role=role)
		return super(TreeValueItem, self).setData(value, role)


	def data(self, role=QtCore.Qt.DisplayRole):
		"""return the right font advance for value text"""
		if role == QtCore.Qt.SizeHintRole:
			return QtCore.QSize(
				len(str(self.tree.value)) * 7.5 + 3,
				rowHeight)
		base = self.processValueFromDisplay(
			super(TreeValueItem, self).data(role))
		base = super(TreeValueItem, self).data(role)
		return base


	def __repr__(self):
		return "<ValueItem {}>".format(self.data())
	def __hash__(self):
		return id(self)

	def onTreeValueChanged(self, branch, oldValue, newValue):
		if branch is not self.tree:
			return
		self.setData(newValue, role=2)



class TreeBranchItem(QtGui.QStandardItem):
	"""small wrapper allowing standardItems to take tree objects directly"""

	def __init__(self, tree):
		""":param tree : Tree"""
		self.treeRef = weakref.ref(tree)
		super(TreeBranchItem, self).__init__(self.tree.name)

		self.setColumnCount(1)

	def getTree(self)->Tree:
		assert self.treeRef() is not None, "tree is dead"
		return self.treeRef()

	@property
	def tree(self):
		return self.getTree()

	def __repr__(self):
		return "<BranchItem {}>".format(self.data())

	def __hash__(self):
		return id(self)

	def valueItem(self)->TreeValueItem:
		"""return the associated value item for this branch"""
		return self.parent().child(self.row(), 1)

	def valueItemClsForBranch(self, branch:Tree):
		"""return the value item class for the given tree branch"""
		return TreeValueItem

	def makeValueItemForBranch(self, branch:Tree):
		"""return a value item for the given tree branch"""
		return self.valueItemClsForBranch(branch)(branch)

	@classmethod
	def itemsForBranch(cls, branch:Tree):
		"""return (TreeBranchItem, TreeValueItem) for the
		given tree branch"""
		# create branches first
		branchItem = cls(branch)
		valueItem = branchItem.makeValueItemForBranch(branch)

		mainItems = (branchItem, valueItem)
		for i in branch.branches:
			branchItems = cls.itemsForBranch(i)
			mainItems[0].appendRow(branchItems)
		#print("itemsForBranch returning", mainItems)
		return mainItems