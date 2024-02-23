from __future__ import annotations

import typing as T

from PySide2 import QtCore, QtGui, QtWidgets

from wplib.constant import LITERAL_TYPES, SEQ_TYPES, MAP_TYPES
from wplib.object import VisitAdaptor
from wptree import TreeInterface
from wpui.superitem import SuperModel, SuperItem

# consder inverting control for selecting new super items or widgets -
# allow representing referencing where needed withut modifying core logic

# node model to separate node graph data from gqraphical representation

class TreeSuperModel(SuperModel):
	"""model for a Tree"""
	forTypes = (TreeInterface, )


# class TreeValueItem(SuperItem):
# 	"""called internally by TreeBranchItem"""
# 	def _generateItemsFromPyObj(self) -> list[tuple[SuperItem, VisitAdaptor.ChildType.T()]]:
# 		"""generate items from pyObj
# 		return dict instead of list here for now,
# 		effect is localised and it's easier to track"""
# 		results = []

class TreeBranchItem(SuperItem):
	"""called internally by TreeSuperItem"""
	forTypes = ()

	wpPyObj : TreeInterface

	def _generateItemsFromPyObj(self) -> list[tuple[SuperItem, VisitAdaptor.ChildType.T()]]:
		"""
		only create new item for value
		"""
		results = []
		nameItemType = self._getComponentTypeForObject(self.wpPyObj.name, "item")
		nameItem = (
			nameItemType(self.wpPyObj.name, wpChildType=VisitAdaptor.ChildType.TreeName,
			              parentQObj=self.wpItemModel,
			              parentSuperItem=self),
			VisitAdaptor.ChildType.TreeName
		)
		results.append(nameItem)

		valueItemType = self._getComponentTypeForObject(self.wpPyObj.value, "item")
		valueItem = (
			valueItemType(self.wpPyObj.value, wpChildType=VisitAdaptor.ChildType.TreeValue,
			              parentQObj=self.wpItemModel,
			              parentSuperItem=self),
			VisitAdaptor.ChildType.TreeValue
		)
		results.append(valueItem)
		return results


class TreeSuperItem(SuperItem):
	"""superitem for a Tree
	difficult here - visitor treats each tree branch in isolation,
	but ui needs to know about the whole tree

	outer representation of a tree - single item per overall tree

	internally uses tree name and value items to separate logic of tree interaction
	from overall system

	on balance, better to have one item per branch, then internally have 2
	sub items - only difficulty is algning values, then later if we want to implement
	drag selection

	"""
	forTypes = (TreeInterface, )

	wpPyObj : TreeInterface


	def _generateItemsFromPyObj(self) -> list[tuple[SuperItem, VisitAdaptor.ChildType.T()]]:
		"""generate items from pyObj
		return dict instead of list here for now,
		effect is localised and it's easier to track
		don't delegate to branch here, since we need 2 items in row for
		appearance to work properly
		"""
		results = []
		for branch in self.wpPyObj.allBranches(includeSelf=True):
			branchItemType : type[TreeBranchItem] = TreeBranchItem
			branchItem = (branchItemType(branch, wpChildType=VisitAdaptor.ChildType.TreeBranch,
				                parentQObj=self.wpItemModel,
				                parentSuperItem=self),
				 VisitAdaptor.ChildType.TreeBranch,

				 )
			# valItemType = self._getComponentTypeForObject(branch.value, "item")
			# valItem = (
			# 	valItemType(branch, wpChildType=VisitAdaptor.ChildType.TreeValue,
			# 	              parentQObj=self.wpItemModel,
			# 	              parentSuperItem=self),
			# 	VisitAdaptor.ChildType.TreeValue
			# )
			# #results[id(branch)] = (branchItem, valItem)
			results.append(branchItem)
			# results.append(valItem)


		return results

	def _insertItemsToModel(self,
	                        items: list[tuple[SuperItem, VisitAdaptor.ChildType.T()]]
	                        #items: dict[int, tuple[QtGui.QStandardItem, VisitAdaptor.ChildType.T()]]
	                        ):
		"""
		track back tree parents to insert items in correct order
		"""
		idBranchMap : dict[int, SuperItem] = {}
		for i in range(len(items)):
			#row = (items[i], items[i+1])
			branch = items[i][0].wpPyObj
			idBranchMap[id(branch)] = items[i][0]
			if branch.parent:
				parentItem = idBranchMap.get(id(branch.parent))
				if parentItem:
					parentItem.insertRow(0, items[i][0])
			else:
				self.wpItemModel.appendRow(items[i][0])



	def wpResultObj(self) ->T.Any:
		"""return the result object
		here just return the attached tree?"""
		return self.wpPyObj.copy()
		results = []
		for i in range(0, len(self.wpChildSuperItems()), 2 ):
			key = self.wpChildSuperItems()[i].wpResultObj()
			value = self.wpChildSuperItems()[i+1].wpResultObj()
			results.append(((key, value), VisitAdaptor.ChildType.MapItem))

		return self.wpVisitAdaptor.newObj(
			self.wpPyObj,
			results
		)
