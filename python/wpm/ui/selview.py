from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import fnmatch

from PySide6 import QtCore, QtWidgets, QtGui

from maya import cmds
import maya.api.OpenMaya as om

"""
now yes this could be a specialisation of a generalisation to filter views
of arbitrary sources

but in the interest of actually doing things, let's revisit that another time


list view live-linked to selection in maya, with filter/search options

be explicit on which functions trigger signals/effects, and which only sync UI

"""


class MayaList(QtWidgets.QWidget):
	"""
	todo: maybe integrate this with the better selectionModel
		system
	"""

	def __init__(self,
	             sourceFn:T.Callable[[], list[str]]=None,
	             contents=None,
	             parent=None):
		super().__init__(parent)
		self.sourceFn = sourceFn
		self._contents = contents
		self._selChanging = False
		self._holdSel = []

	def makeWidgets(self):
		self.filterLine = QtWidgets.QLineEdit(self)
		self.filterLine.setPlaceholderText("filter")

		self.list = QtWidgets.QListWidget(self)
		self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

		self.syncBox = QtWidgets.QCheckBox("Sync", self)


	def makeLayouts(self):
		pass

	def makeSignals(self):
		self.syncBox.clicked.connect(self.onSyncBoxChanged)
		self.list.selectionChanged.connect(
			self.onSelectionChanged
		)
		pass

	def repopulate(self):
		"""rerun source function, re-filter, re-select"""
		self._holdSel = self.getSelection()
		if self.sourceFn is not None:
			newItems = self.sourceFn()
		else:
			newItems = self._contents
		newItems = self.applyFilter(newItems)
		self.list.clear()
		for i in newItems:
			self.list.addItem(i)
		self.setSelection(self._holdSel)

	def getSelection(self, filtered=False)->list[str]:
		sel = [i.text() for i in self.list.selectedItems()]
		if not filtered:
			return sel
		return self.applyFilter(sel)

	def setSelection(self, names:list[str]=()):
		"""todo: selection modes to add, replace, invert, etc"""
		self.list.clearSelection()
		for i in names:
			for item in self.list.findItems(i, QtCore.Qt.MatchFixedString):
				self.list.setCurrentItem(item)

	def getFilterStr(self):
		return self.filterLine.text()

	def setFilterStr(self, text):
		self.filterLine.setText(text)

	def applyFilter(self, items:list[str])->list[str]:
		return list(fnmatch.filter(items, self.getFilterStr()))

	def onFilterStrChanged(self, *_, **__):
		self.repopulate()

	def onMayaSelChanged(self, *_, **__):
		"""called whenever Maya selection changes"""
		if self._selChanging: # already in course of changing selection
			return
		self._selChanging = True
		mayaSel = set(cmds.ls(sl=1) or [])
		self.setSelection(mayaSel)

		self._selChanging = False

	def onSelectionChanged(self):
		if self._selChanging:
			return
		self._selChanging = True

		cmds.select(cl=1)
		for i in self.getSelection():
			if cmds.objExists(i):
				cmds.select(i, add=1)

		self._selChanging = False

	def selectionIsSynced(self):
		return self.syncBox.isChecked()

	def onSyncBoxChanged(self, *_, **__):
		""""""
		self.repopulate()

	def setContents(self, contents:list[str]):
		""""""
		self._contents = contents
		self.repopulate()


