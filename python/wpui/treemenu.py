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
	# log("buildMenu", tree)
	# log(tree.branchMap())
	for name, branch in tree.branchMap().items():
		if branch.branches: # make new menu
			newMenu = menu.addMenu(branch.name)
			if branch.auxProperties.get("icon"):
				menu.setIcon(branch.auxProperties["icon"])
			buildMenuFromTree(branch, newMenu)
			continue
		else :
			action = menu.addAction(branch.name)
			log("newAction", action, branch.name, branch.v, branch.auxProperties)
			# multiple functions under same branch
			if isinstance(branch.value, (list, tuple)):
				action.triggered.connect(lambda: EVAL(i) for i in branch.value)
			# single function, just trigger as normal
			elif branch.value is not None:
				action.triggered.connect(lambda : EVAL(branch.value))

			if branch.auxProperties.get("icon"):
				action.setIcon(branch.auxProperties["icon"])
			if "enable" in branch.auxProperties: # allow greying out some actions
				if not EVAL(branch.auxProperties["enable"]):
					action.setEnabled(False)
	return menu

def collateMenuTrees(trees:T.Iterable[Tree], parentBranch:Tree=None):
	"""
	TODO:
	if 2 branches of the same name are found, and they both have callable values,
	merge them together"""


base = object
if T.TYPE_CHECKING:
	base = QtWidgets.QWidget
class ContextMenuProvider(base):
	"""mixin to add a tree for sorting
	context menu actions

	TODO: consider logic for delegating to parents? adding
		options to the context menu of parents?
	"""

	def __init__(self, *args, **kwargs):
		self._contextMenuWidget : QtWidgets.QMenu = None # keeps live reference to menu when shown
		self._contextMenuTree : Tree = self._getBaseContextTree(*args, **kwargs)

	def _getBaseContextTree(self, *args, **kwargs)->Tree[str, callable]:
		"""passed args and kwargs from init"""
		return Tree("contextMenuTree")

	def _getContextTreeForEvent(self, event:QtGui.QContextMenuEvent):
		"""called on each event - if you need to query the wider state
		to show context menu actions, do it here -
		return a copy of the base tree if you modify it"""
		return self._contextMenuTree

	def _getBaseQtContextMenu(self)->QtWidgets.QMenu:
		""" OVERRIDE
		return a default qt context menu, if it exists"""
		try: return self.createStandardContextMenu()
		except AttributeError: return None

	def contextMenuEvent(self, event:QtGui.QContextMenuEvent):
		baseMenu = self._getBaseQtContextMenu()
		menuTree = self._getContextTreeForEvent(event)
		treeMenu = buildMenuFromTree(menuTree, menu=baseMenu)
		self._contextMenuWidget = treeMenu
		#treeMenu.aboutToHide.connect(lambda *a, **k : setattr(self, "_contextMenuWidget", None))
		treeMenu.move(event.globalPos())
		treeMenu.show()







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


