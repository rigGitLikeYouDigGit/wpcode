

from __future__ import annotations
import typing as T

from wptree.interface import TreeInterface
from wptree.ui.model import TreeModel
from wptree.ui.view import TreeView

from wpdex.ui.superitem import SuperItem

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



