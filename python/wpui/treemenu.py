from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wptree import Tree #todo: tree generators
from wpdex import EVAL
from PySide2 import QtCore, QtWidgets, QtGui


def buildMenuFromTree(
		tree:Tree,
		menu:QtWidgets.QMenu=None
)->QtWidgets.QMenu:
	"""expect tree with values of partials or lambdas"""
	menu = menu or QtWidgets.QMenu(title=tree.name) # top level title never appears
	log("buildMenu", tree)
	log(tree.branchMap())
	for name, branch in tree.branchMap().items():
		if branch.branches: # make new menu
			newMenu = menu.addMenu(branch.name)
			if branch.auxProperties.get("icon"):
				menu.setIcon(branch.auxProperties["icon"])
			buildMenuFromTree(branch, newMenu)
			continue
		else :
			action = menu.addAction(branch.name)
			log("newAction", action, branch.name, branch.auxProperties)
			if branch.value is not None:
				action.triggered.connect(lambda : EVAL(branch.value))
			if branch.auxProperties.get("icon"):
				action.setIcon(branch.auxProperties["icon"])
			if "enable" in branch.auxProperties: # allow greying out some actions
				if not EVAL(branch.auxProperties["enable"]):
					action.setEnabled(False)
	return menu





class _TreeMenu(QtWidgets.QMenu):
	"""from older version as reference, delete once everything's solid"""
	def __init__(self, parent:QtWidgets.QWidget=None, tree=None,
	             title="Menu",
	             treeExtraKey="qmenu"):
		super(_TreeMenu, self).__init__(title, parent)
		tree = tree or Tree("menu")
		self.tree : Tree = None
		self.setTree(tree)


	def setTree(self, tree:Tree):
		self.tree = tree
		self.tree.lookupCreate = True
		self.buildMenuFromTree(self.tree, self)

	def refresh(self):
		self.clear()
		self.buildMenuFromTree(self.tree, self)

	def onContextRequested(self):
		"""to be used if this menu is a context menu"""
		self.buildMenuFromTree(self.tree, self)

	@classmethod
	def branchesMatchForOptions(cls, branches:list[Tree]):
		"""if all branches have compatible sets of options, return True
		else return False (most common)"""
		baseKeys = set(optionMapFromOptions(branches[0].options).keys())
		for branch in branches[1:]:
			keys = set(optionMapFromOptions(branch.options).keys())
			if keys != baseKeys:
				return False
		return baseKeys

	@classmethod
	def optionActionsForBranches(cls, parentMenuBranch:Tree, optionBranches:list[Tree]):
		"""add actions for each option available for tree value
		we already guarantee that branches share the same available options"""
		optionMap = optionMapFromOptions(optionBranches[0].options)
		for k, v in optionMap.items():
			# create lambda function, add it to parent tree
			def _setVals(*args, **kwargs):
				for targetBranch in optionBranches:
					# last failsafe to avoid setting invalid options
					branchOptionMap = optionMapFromOptions(targetBranch.options)
					if k in branchOptionMap:
						targetBranch.v = branchOptionMap[k]
			parentMenuBranch(k).v = _setVals


	@classmethod
	def buildMenuFromTree(cls, tree:Tree, parentMenu:QtWidgets.QMenu=None):
		"""returns qmenu corresponding to tree"""
		parentMenu = parentMenu or QtWidgets.QMenu(title=tree.name)
		parentMenu.clear()
		for i in tree.branches:
			added = None
			if i.branches:
				added = parentMenu.addMenu(TreeMenu.buildMenuFromTree(i))
			elif isinstance(i.value, PartialAction):
				i.value.setParent(parentMenu)
				added = parentMenu.addAction(i.value)
			elif callable(i.value):
				added = parentMenu.addAction(i.name)
				added.triggered.connect(i.value)
			# check for list of multiple actions
			elif isinstance(i.value, (list, tuple)):
				added = parentMenu.addAction(i.name)
				for slot in i.value:
					added.triggered.connect(slot)

			if added is None:
				#tree.display()
				raise RuntimeError("no action value found for branch {}".format(i))

			if i.description:
				added.setToolTip(i.description)
		return parentMenu


