
from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets

if T.TYPE_CHECKING:
	from wptree import Tree
	from wptree.ui.model import TreeModel


class TreeView(QtWidgets.QTreeView):
	"""relatively thin viewer for a tree Qt model.
	A tree view holds only a path to its focused branch,
	and a reference to the shared tree model it's viewing.
	"""

	def __init__(self, parent=None):
		super(TreeView, self).__init__(parent)
		self.rootPath : list[str] = None

	if T.TYPE_CHECKING:
		def model(self) -> TreeModel:
			pass

	def setRootBranch(self, branch:Tree):
		"""set the root branch of the tree view.
		"""
		self.rootPath = branch.address(includeSelf=True, includeRoot=False)

