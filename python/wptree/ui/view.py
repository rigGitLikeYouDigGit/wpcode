
from __future__ import annotations
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets

from wpui.keystate import KeyState
from wpui.constant import keyDict, dropActionDict, tabKeys, enterKeys, deleteKeys, shiftKeys, escKeys, spaceKeys
from wpui.event import AllEventEater



from wptree.ui.constant import relAddressRole, treeObjRole, addressRole
from wptree.ui.model import TreeModel

from wpui.superitem import SuperItem, SuperModel, SuperDelegate, SuperViewBase

if T.TYPE_CHECKING:
	from wptree import Tree
	from wptree.ui.item import TreeBranchItem

expandingPolicy = QtWidgets.QSizePolicy(
	QtWidgets.QSizePolicy.Expanding,
	QtWidgets.QSizePolicy.Expanding,
)

class FocusEventEater(QtCore.QObject):
	def eventFilter(self, watched:QtCore.QObject,
					event:QtCore.QEvent):
		if not any([event.type() == i for i in [
			QtCore.QEvent.FocusIn,
			QtCore.QEvent.FocusOut,

		]]):
			return QtCore.QObject.eventFilter(self, watched, event)
		#print("stopped", event)
		return True


class TreeView( SuperViewBase, QtWidgets.QTreeView,):
	"""relatively thin viewer for a tree Qt model.
	A tree view holds only a path to its focused branch,
	and a reference to the shared tree model it's viewing.
	"""
	currentBranchChanged = QtCore.Signal(dict)

	def __init__(self, parent=None):
		#super(TreeView, self).__init__(parent)
		QtWidgets.QTreeView.__init__(self, parent)
		SuperViewBase.__init__(self)
		self.rootPath : list[str] = None

		#self.installEventFilter(AllEventEater(self))


		self.keyState = KeyState()

		self.savedCollapsedUids : set[str] = set()
		self.savedSelection = []
		self.midUiOperation = 0
		self.scrollPos = 0

		self.makeBehaviour()
		self.makeAppearance()
		self.setSelectionBehavior(self.SelectRows)
		#self.setSelectionMode(self.ExtendedSelection)
		self.setEditTriggers(self.DoubleClicked)

		# self.setSizeAdjustPolicy(
		# 	QtWidgets.QAbstractScrollArea.AdjustToContents
		# )


		#self.setFocusPolicy(QtCore.Qt.ClickFocus)

	# def sizeHintForIndex(self, index:QtCore.QModelIndex) -> QtCore.QSize:
	# 	"""override to return a fixed size for all rows"""
	# 	return QtWidgets.QTreeView.sizeHintForIndex(self, index)

	def onIndexCollapsed(self, index:QtCore.QModelIndex):
		self.savedCollapsedUids.add(self.model().itemFromIndex(index).uid)

	def onIndexExpanded(self, index:QtCore.QModelIndex):
		self.savedCollapsedUids.discard(self.model().itemFromIndex(index).uid)

	def focusNextPrevChild(self, next:bool) -> bool:
		"""override to prevent focus from leaving the tree view.
		"""
		return False

	def makeAppearance(self):


		header = self.header()
		header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
		# header.setStretchLastSection(True)
		#
		# self.setSizeAdjustPolicy(
		# 	QtWidgets.QAbstractScrollArea.AdjustToContents
		# )
		self.setIndentation(10)
		self.setAlternatingRowColors(True)
		self.showDropIndicator()

	def makeBehaviour(self):
		self.setDragEnabled(True)
		self.setAcceptDrops(True)
		self.setDragDropMode(
			QtWidgets.QAbstractItemView.InternalMove
		)
		#self.setSelectionMode(self.ExtendedSelection)
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

		set up all tree-specific stuff, then add superItem
		widgets
		"""
		QtWidgets.QTreeView.setModel(self, model)
		SuperViewBase.setModel(self, model)
		self.model().treeSet.connect(self.onTreeSet)
		self.model().beforeItemChanged.connect(self.onBeforeItemChanged)
		self.model().afterItemChanged.connect(self.onAfterItemChanged)
		self.onTreeSet(model.tree)
		assert isinstance(self.model(), TreeModel)

		#self.setItemDelegate(SuperDelegate(self))
		#print("set model", model)
		#print("items", model.rowCount())

	def onBeforeItemChanged(self, *args, **kwargs):
		self.saveAppearance()

	def onAfterItemChanged(self, *args, **kwargs):
		self.restoreAppearance()

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
		#print("key press", event.key(), event.key() in tabKeys)
		if event.key() in tabKeys:
			for index in self.selectedRowIndices():#type:QtCore.QModelIndex
				#print("index", index, self.model().branchFromIndex(index))
				if self.keyState.shift: #unparent one level
					parentIndex = index.parent()
					grandParentIndex = parentIndex.parent()
					if not grandParentIndex.isValid():
						continue
					self.model().parentRows([index], grandParentIndex, parentIndex.row() + 1)
				else: # parent to previous sibling - if first, do nothing
					if not index.row():
						continue
					targetParent = index.sibling(index.row()-1, 0)
					self.model().parentRows([index], targetParent)
			event.accept()
			return

		if event.key() == QtCore.Qt.Key_Delete: # delete
			for index in self.selectedRowIndices():
				self.model().deleteRow(index)

		if event.key() == QtCore.Qt.Key_P: #reparent
			if self.keyState.shift:
				self.model().parentRows(self.selectedRowIndices(), self.model().index(0,0))
			else:
				self.model().parentRows(self.selectedRowIndices()[:-1], self.selectedRowIndices()[-1])

		if event.key() == QtCore.Qt.Key_D and self.keyState.ctrl: # duplicate
			newTrees = self.model().duplicateRows(self.selectedRowIndices())



		if event.key() == QtCore.Qt.Key_Return: # edit branch
			if self.keyState.shift:
				self.edit(self.model().index(
					self.currentIndex().row(), 0, self.currentIndex().parent()))
			else:
				self.edit(self.model().index(
					self.currentIndex().row(), 1, self.currentIndex().parent()))


		super(TreeView, self).keyPressEvent(event)

	def keyReleaseEvent(self, event:PySide2.QtGui.QKeyEvent) -> None:
		"""manage key state"""
		self.keyState.keyReleased(event)
		super(TreeView, self).keyReleaseEvent(event)

	def mouseDoubleClickEvent(self, event:PySide2.QtGui.QMouseEvent) -> None:
		"""manage key state"""
		#self.keyState.mouseDoubleClicked(event)
		if event.button() == QtCore.Qt.RightButton: #double right click get out of here
			return
			return super(TreeView, self).mouseDoubleClickEvent(event)
		if event.button() == QtCore.Qt.LeftButton:
			self.edit(self.currentIndex())
			return
		return super(TreeView, self).mouseDoubleClickEvent(event)

	def mouseReleaseEvent(self, event:PySide2.QtGui.QMouseEvent) -> None:
		"""manage key state"""
		self.keyState.mouseReleased(event)
		super(TreeView, self).mouseReleaseEvent(event)

	def dragEnterEvent(self, event:PySide2.QtGui.QDragEnterEvent) -> None:
		return

	def dragMoveEvent(self, event:PySide2.QtGui.QDragMoveEvent) -> None:
		return

	def dragLeaveEvent(self, event:PySide2.QtGui.QDragLeaveEvent) -> None:
		return

	def mouseMoveEvent(self, event:PySide2.QtGui.QMouseEvent) -> None:
		return

	def mousePressEvent(self, event):
		"""manage key state"""
		self.keyState.mousePressed(event)

		# only pass event on editing,
		# need to manage selection separately
		if event.button() == QtCore.Qt.RightButton:
			return super(TreeView, self).mousePressEvent(event)

		# indexAt only checks the whole row -
		# to expand based on icon, need to check the visual rect
		index = self.indexAt(event.pos())
		rect = self.visualRect(index)
		if not rect.contains(event.pos()):
			# clicked on expand icon, not actual item
			return super(TreeView, self).mousePressEvent(event)

		#return super(TreeView, self).mousePressEvent(event)

		self.onClicked(index)
		self.ensureRowsSelected()
		#event.accept()
		#print("selected", self.selectionModel().selectedIndexes())

		return


	def ensureRowsSelected(self):
		"""run after selection change to ensure rows are selected"""

		# get full rows from selected row indices (output from OnClicked)
		selIndices = []
		indices = self.selectionModel().selectedIndexes()
		for i in indices:

			if i.column() == 0:
				selIndices.append(i)
		self.selectionModel().clearSelection()
		for rowIndex in selIndices:
			for itemIndex in self.indicesForRowIndex(rowIndex):
				self.selectionModel().select(itemIndex, QtCore.QItemSelectionModel.Select)

	def pickWalkFromRows(self, rows:list[QtCore.QModelIndex], key,
	                     stickOnEndOfRows=True)->list[QtCore.QModelIndex]:
		"""return selection of new rows based on key press
		if stickOnEndOfRows, will return last entry in row if already
		selected - if not, will return nothing
		"""
		newRows = []
		if key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right,
			QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):

			for i in rows:
				adj = self.model().connectedIndices(i)
				target = None
				if key == QtCore.Qt.Key_Left:
					# back one index
					target = adj["prev"]
				elif key == QtCore.Qt.Key_Right:
					# forwards one index
					target = adj["next"]
				elif key == QtCore.Qt.Key_Up:
					# up to parent
					target = adj["parent"]
				elif key == QtCore.Qt.Key_Down:
					target = adj["child"]

				if target:
					newRows.append(target)
				elif stickOnEndOfRows: # leave branch selected if no target
					newRows.append(i)

		return newRows


	def onClicked(self, index:QtCore.QModelIndex):
		""" manage selection manually
		selection only ever works by rows"""

		#rowIndices = self.indicesForRowIndex(index)
		#rowSel = QtCore.QItemSelection()
		# for row in rowIndices:
		# 	rowSel.select(row, row)

		baseIndex = index

		index = self.model().index(index.row(), 0, index.parent())

		if not self.keyState.shift:
			# no contiguous selection

			#print("key state", self.keyState.ctrl, self.keyState.debug())

			if self.keyState.ctrl:
				command = QtCore.QItemSelectionModel.Toggle
			elif self.keyState.alt:
				print("deselect")
				command = QtCore.QItemSelectionModel.Deselect
			else:
				#self.selectionModel().clear()
				command = QtCore.QItemSelectionModel.ClearAndSelect

			self.selectionModel().select(
				index,
				command
			)
			# self.selectionModel().setCurrentIndex(
			# 	index, QtCore.QItemSelectionModel.Current
			# )
			# return

		elif self.keyState.shift: # contiguous span

			clickRow = self.model().rowFromIndex(index)
			if not self.selectionModel().currentIndex().isValid():
				self.selectionModel().setCurrentIndex(
					index, QtCore.QItemSelectionModel.Current
				)
				self.selectionModel().select(
					index,
					QtCore.QItemSelectionModel.Select
				)
				return
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
			while checkIdx != clickRow: #and count < 4:
				count += 1
				checkIdx = fn(checkIdx)
				if not checkIdx.isValid():
					break

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
			baseIndex,
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

	def indicesForRowIndex(self, index:QtCore.QModelIndex)->T.List[QtCore.QModelIndex]:
		"""returns all indices in the same row as the given index"""
		indices = []
		for i in range(self.model().columnCount()):
			indices.append(self.model().index(index.row(), i, index.parent()))
		return indices

	def selectedRowIndices(self)->list[QtCore.QModelIndex]:
		"""returns branch indices for all name and value items selected in ui"""
		indices = []
		for i in self.selectionModel().selectedIndexes():
			row = self.model().index(i.row(), 0, i.parent())
			if row in indices:
				continue
			indices.append(row)
		return indices


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

	def saveCollapsedBranches(self):
		for i in self.model().allRows():
			if not self.isExpanded(i):
				self.savedCollapsedUids.add(
					self.model().itemFromIndex(i).uid
				)

	def restoreCollapsedBranches(self):
		""" restores saved expansion state """
		from wptree.ui.item import TreeBranchItem

		for uid in self.savedCollapsedUids:
			item = TreeBranchItem.getByIndex(uid)
			self.expand(item.index())


	def saveAppearance(self, *args, **kwargs):
		""" saves expansion and selection state
		if clear, will remove any previously saved trees
		"""
		#print("save appearance", self.midUiOperation)
		self.midUiOperation += 1

		if self.midUiOperation > 1: # don't do anything if visual transaction already working
			return
		#
		# if clear:
		# 	self.savedSelectedTrees.clear()
		# 	self.savedCollapsedTrees.clear()
		# self.currentSelected = None
		# for i in self.selectionModel().selectedRows():
		# 	branch = self.model().treeFromRow(i)
		# 	self.savedSelectedTrees.append(branch)
		# for i in self.model().allRows():
		# 	if not self.model().checkIndex(i):
		# 		print("index {} is not valid, skipping".format(i))
		# 	if not self.isExpanded(i):
		# 		branch = self.model().treeFromRow(i)
		# 		if branch:
		# 			self.savedCollapsedTrees.append(branch)
		# if self.selectionModel().currentIndex().isValid():
		# 	self.currentSelected = self.model().treeFromRow(
		# 		self.selectionModel().currentIndex() )
		self.saveCollapsedBranches()
		# save viewport scroll position
		self.scrollPos = self.verticalScrollBar().value()

	def restoreAppearance(self, *args, **kwargs):
		""" restores expansion and selection state """
		#print("restore appearance", self.midUiOperation)

		self.midUiOperation -= 1
		if self.midUiOperation:
			return
		#self.setRootIndex(self.model().invisibleRootItem().child(0, 0).index())

		# self.resizeToTree()

		#print("saved selected", self.savedSelectedTrees)
		# for i in self.savedSelectedTrees:
		# 	#print("check ", i)
		# 	if not self.model().rowFromTree(i):
		# 		#print("no saved branch found for ", i)
		# 		continue
		#
		# 	self.selectionModel().select(
		# 		self.model().rowFromTree(i),
		# 		QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows
		# 	)
		# trees default to expanded if not found
		self.expandAll()
		# for i in self.savedCollapsedTrees:
		# 	if not self.model().rowFromTree(i):
		# 		#print("expanded tree not found ")
		# 		continue
		# 	self.collapse( self.model().rowFromTree(i) )
		# 	pass
		# if self.currentSelected: # on deletion need to fix up this reference
		# 	#print("setting current")
		# 	row = self.model().rowFromTree(self.currentSelected)
		# 	if row is not None:
		# 		self.sel.setCurrent(row)
		self.verticalScrollBar().setValue(self.scrollPos)
		#
		# #self.resizeToTree()
		# self.savedSelectedTrees.clear()
		# self.savedCollapsedTrees.clear()


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

