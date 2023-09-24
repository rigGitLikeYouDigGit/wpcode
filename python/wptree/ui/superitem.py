

from __future__ import annotations
import typing as T
import os, pathlib, weakref

from PySide2 import QtGui, QtCore
from wplib import log
from wplib.constant import LITERAL_TYPES

from wplib.object import UidElement

from wptree.interface import TreeInterface
from wptree.main import Tree
from wptree.delta import TreeDeltas
from wptree.ui.constant import addressRole, relAddressRole, childBoundsRole, treeObjRole, rowHeight
from wptree.ui.view import TreeView
from wptree.ui.item import TreeValueItem, TreeBranchItem
from wptree.ui.model import TreeModel
from wptree.ui.view import TreeView

from wpui.superitem.base import SuperItem, SuperModel, SuperDelegate


if T.TYPE_CHECKING:
	from tree.ui.model import TreeModel


class TreeSuperItem(SuperItem):
	if T.TYPE_CHECKING:
		modelCls = TreeModel
	viewCls = TreeView
	forCls = TreeInterface


	# superItem integration
	# def makeChildItems(self):
	# 	"""return a list of child items for this branch -
	# 	will probably only be called directly for root item
	# 	and value items"""
	# 	return self.itemsForBranch(self.superItemValue)

	# def getNewChildModel(self):
	# 	"""return a new model for this branch"""
	# 	print("tree branch get new child model")
	# 	treeModel : TreeModel = super(TreeBranchItem, self).getNewChildModel()
	# 	treeModel.setTree(self.superItemValue)
	# 	return treeModel

	# @classmethod
	# def _getNewInstanceForValue(cls, value) ->SuperItem:
	# 	"""return a new instance of this class for the given value"""
	# 	return cls(value)


	def setValue(self, value):
		"""exclude from wider superItem setting for now -
		branch items are not editable,
		and treeRef should be only handle on tree"""
		# #self.sync()
		# return
		super(TreeSuperItem, self).setValue(value)
		self.childModel.setTree(self.superItemValue)
		# self.treeRef = weakref.ref(value)
		# self.model().setTree(self.superItemValue)
		# pass

	# def getNewView(self) ->viewCls:
	# 	"""return a new view for this branch"""
	# 	print("tree branch get new view")
	# 	print(self.superItemValue)
	# 	print(self.childModel)
	# 	return super(TreeSuperItem, self).getNewView()



