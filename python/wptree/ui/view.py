
from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets

from wpui.keystate import KeyState

from wptree.ui.constant import relAddressRole, treeObjRole, addressRole

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
	currentBranchChanged = QtCore.Signal(dict)

	def __init__(self, parent=None):
		super(TreeView, self).__init__(parent)
		self.rootPath : list[str] = None

		self.keyState = KeyState()

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

	#region events
	def keyPressEvent(self, event:PySide2.QtGui.QKeyEvent) -> None:
		""" bulk of navigation operations,
		for hierarchy navigation aim to emulate maya outliner

		ctrl+D - duplicate
		del - delete

		left/right - select siblings
		up / down - select child / parent

		p - parent selected branches to last selected
		shiftP - parent selected branches to root

		alt + left / right - shuffle selected among siblings

		not sure if there is an elegant way to structure this
		going with disgusting battery of if statements

		for selection, expansion etc:
		shift adds
		control removes
		shift + control toggles

		alt makes actions recursive?
		"""
		self.keyState.keyPressed(event)
		super(TreeView, self).keyPressEvent(event)

	def keyReleaseEvent(self, event:PySide2.QtGui.QKeyEvent) -> None:
		"""manage key state"""
		self.keyState.keyReleased(event)
		super(TreeView, self).keyReleaseEvent(event)

	def mousePressEvent(self, event):
		"""manage key state"""
		self.keyState.mousePressed(event)

		# only pass event on editing,
		# need to manage selection separately
		if event.button() == QtCore.Qt.RightButton:
			return super(TreeView, self).mousePressEvent(event)


		# print("mouse press", )
		# self.keyState.debug()
		index = self.indexAt(event.pos())
		self.onClicked(index)
		event.accept()


	def onClicked(self, index):
		""" manage selection manually """
		if not self.keyState.shift:
			# no contiguous selection
			command = QtCore.QItemSelectionModel.ClearAndSelect
			if self.keyState.ctrl:
				command = QtCore.QItemSelectionModel.Toggle
			if self.keyState.alt:
				command = QtCore.QItemSelectionModel.Deselect
			self.selectionModel().select(
				index,
				command
			)
		elif self.keyState.shift: # contiguous span

			clickRow = self.model().rowFromIndex(index)
			currentRow = self.model().rowFromIndex(
				self.selectionModel().currentIndex()
			)
			# find physically lowest on screen
			if self.visualRect(clickRow).y() < \
				self.visualRect(currentRow).y():
				fn = self.indexAbove
			else:
				lowest = clickRow
				highest = currentRow
				fn = self.indexBelow
			# if it's stupid and it works, what's the point of being clever
			targets = []
			selStatuses = []
			checkIdx = currentRow
			selRows = self.selectionModel().selectedRows()
			count = 0
			while checkIdx != clickRow and count < 4:
				count += 1
				checkIdx = fn(checkIdx)

				targets.append(checkIdx)
				selStatuses.append(checkIdx in selRows)

			# this was some wild logic - go back to simple toggle,
			# add or remove
			# addOrRemove = sum(selStatuses) < len(selStatuses) / 2
			# for row in targets:
			# 	self.selectionModel().select(
			#
			# 		self.model().index(row, 0),
			# 		QtCore.QItemSelectionModel.Select
			# 	)
			# 	self.sel.add(row)

			operation = QtCore.QItemSelectionModel.Toggle
			if self.keyState.ctrl:
				operation = QtCore.QItemSelectionModel.Select
			if self.keyState.alt:
				operation = QtCore.QItemSelectionModel.Deselect

			for row in targets:
				self.selectionModel().select(
					self.model().rowFromIndex(row),
					operation
				)

		# set previous selection
		self.selectionModel().setCurrentIndex(
			index,
			QtCore.QItemSelectionModel.Current
		)

	def onCurrentChanged(self,
	                     currentIdx:QtCore.QModelIndex,
	                     prevIdx:QtCore.QModelIndex):
		"""connected to selection model - convert model indices
		to branches, then emit top-level signal"""
		newBranch = currentIdx.data(treeObjRole)
		prevBranch = prevIdx.data(treeObjRole)
		self.currentBranchChanged.emit(
			{"oldBranch" : prevBranch,
			 "newBranch" : newBranch}
		)


	# end region

	def selectedBranches(self)->T.List[Tree]:
		"""returns branches for all name and value items selected in ui"""
		branchList = []
		for i in self.selectionModel().selectedRows():
			branch = self.model().branchFromIndex(i)
			if branch in branchList:
				continue
			branchList.append(branch)
		return branchList

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

