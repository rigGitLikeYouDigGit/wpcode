from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets

from wptree import Tree
from wptree.ui.view import TreeView
from wptree.ui.model import TreeModel

"""strict convenience class, no extra functionality"""


class TreeWidget(QtWidgets.QWidget):

	def __init__(self, parent=None, tree:Tree=None):
		super(TreeWidget, self).__init__(parent)
		self.view = TreeView(parent=self)
		self.model : TreeModel = None

		vl = QtWidgets.QVBoxLayout()
		vl.addWidget(self.view)
		self.setLayout(vl)

		if tree:
			self.setTree(tree)

	def setTree(self, tree:Tree):
		print("setting tree", tree)
		self.model = TreeModel.modelForTree(tree, parent=None)
		self.view.setModel(self.model)
		self.view.setRootBranch(tree.root)


