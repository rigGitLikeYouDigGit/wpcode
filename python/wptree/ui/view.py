
from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets

if T.TYPE_CHECKING:
	from wptree import Tree
	from wptree.ui.model import TreeModel

expandingPolicy = QtWidgets.QSizePolicy(
	QtWidgets.QSizePolicy.Expanding,
	QtWidgets.QSizePolicy.Expanding,
)

class TreeView(QtWidgets.QTreeView):
	"""relatively thin viewer for a tree Qt model.
	A tree view holds only a path to its focused branch,
	and a reference to the shared tree model it's viewing.
	"""

	def __init__(self, parent=None):
		super(TreeView, self).__init__(parent)
		self.rootPath : list[str] = None

		self.makeBehaviour()
		self.makeAppearance()

	def makeAppearance(self):


		header = self.header()
		header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
		header.setStretchLastSection(True)

		self.setSizeAdjustPolicy(
			QtWidgets.QAbstractScrollArea.AdjustToContents
		)
		self.setUniformRowHeights(True)
		self.setIndentation(10)
		self.setAlternatingRowColors(True)
		self.showDropIndicator()

	def makeBehaviour(self):
		self.setDragEnabled(True)
		self.setAcceptDrops(True)
		self.setDragDropMode(
			QtWidgets.QAbstractItemView.InternalMove
		)
		self.setSelectionMode(self.ExtendedSelection)
		self.setSelectionBehavior(self.SelectRows)
		#self.setDragDropMode()
		#self.setDropIndicatorShown()
		self.setAutoScroll(False)
		self.setFocusPolicy(QtCore.Qt.ClickFocus)
		self.setDefaultDropAction(QtCore.Qt.CopyAction)


	if T.TYPE_CHECKING:
		def model(self) -> TreeModel:
			pass

	def setModel(self, model:TreeModel):
		"""set the tree model for this view.
		"""
		super(TreeView, self).setModel(model)
		self.model().treeSet.connect(self.onTreeSet)
		self.onTreeSet(model.tree)

	def setRootBranch(self, branch:Tree):
		"""set the root branch of the tree view.
		"""
		self.rootPath = branch.address(includeSelf=True, includeRoot=False)

	def onTreeSet(self, tree:Tree):
		"""called when the tree model is set.
		"""
		self.expandAll()

	# region visuals
	def boundingRectForBranch(self, branch, includeBranches=True)->QtCore.QRect:
		index = self.model().rowFromTree(branch)
		rect = self.visualRect(index)
		if includeBranches:
			for i in self.model().allChildren(index):
				rect = rect.united(self.visualRect(i))
		return rect

	def showValues(self, state=True):
		"""tracks if value column is shown or not"""
		self.setColumnHidden(1, not state)

	def valuesShown(self)->bool:
		return not self.isColumnHidden(1)

	# endregion visuals

	# region copy/paste, drag/drop

	def copyEntries(self):
		clip = QtGui.QGuiApplication.clipboard()
		indices = self.selectionModel().selectedRows()
		if not indices: # nothing to copy
			return
		mime = self.model().mimeData(indices)
		clip.setMimeData(mime)

	def pasteEntries(self):
		indices = self.selectedIndexes()  # i strongly hate
		if not indices:
			return
		index = indices[0]
		clip = QtGui.QGuiApplication.clipboard()
		data = clip.mimeData()
		self.model().dropMimeData(data,
								  QtCore.Qt.CopyAction,
								  0,
								  0,
								  index)

