from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtWidgets, QtGui

from wpdex.ui import AtomicWindow, AtomicStandardItemModel, AtomicView
from wpui.treemenu import ContextMenuProvider
from wpui.widget import Collapsible
from wpdex.ui.treeitem import TreeBranchItem, TreeDexView, TreeDexModel

if T.TYPE_CHECKING:
	from ..attr import NodeAttrRef, NodeAttrWrapper
	from ..node import ChimaeraNode

"""chimaera special dispensation for displaying full
resolved attribute trees,
outlining override values, etc
"""

class AttrWindow(QtWidgets.QFrame):
	"""
	window to display the current content of an attribute -
	but HOW??
	currently we only display the override tree - SHOULD we have
	a second window on the side to show the final overridden
	tree?

	as a first pass it's easier than trying to track coloured boxes
	all over the tree widget to show differences
	"""
	forTypes = ()
	def __init__(self, value:NodeAttrWrapper, parent=None):
		# QtWidgets.QWidget.__init__(self, parent)
		QtWidgets.QFrame.__init__(self, parent)

		self.setLayout(QtWidgets.QHBoxLayout(self))
		# self.model = TreeDexModel(value.override(),
		#                           parent=self)
		self.attr = value
		self.view = TreeDexView(value=value.override(),
		                        parent=self)

		self.layout().addWidget(Collapsible(title=value.name(),
		                                    w=self.view,
		                                    parent=self))

		# add button to show resolved printout of tree
		self.resultView = QtWidgets.QTextEdit(parent=self)
		self.resultViewCollapse = Collapsible(title="result",
		                                    w=self.resultView,
		                                    parent=self)
		self.resultViewCollapse.expandedSignal.connect(self.syncResult)
		self.layout().addWidget(self.resultViewCollapse)

		self.setContentsMargins(0, 0, 0, 0)
		self.layout().setContentsMargins(0, 0, 0, 0)
		self.layout().setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

		ContextMenuProvider.__init__(self, value=value, parent=parent)

	def node(self)->ChimaeraNode:
		return self.attr.node

	def syncResult(self, *args, **kwargs):
		self.resultView.setPlainText(
			self.node().resolveAttribute(self.attr).displayStr()
		)
