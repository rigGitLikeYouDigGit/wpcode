from __future__ import annotations
import typing as T

import os, sys

from pathlib import Path

from PySide2 import QtWidgets, QtCore, QtGui

from dataclasses import dataclass

from tree.lib.object import TypeNamespace
from tree import Tree

from wp import constant as const
from wp.treefield import TreeField
from wp.ui.layout import autoLayout
from wp.ui import delete
from wp.ui.widget import WpWidgetBase, BorderFrame
from wp.ui.treefieldwidget import TreeFieldParams, StringTreeWidget
from wpm import cmds, om, oma, WN, createWN, getSceneGlobals

from wpm.lib.ui import window
from wpm.lib.ui.nodetracker import NodeTracker
from .. import lib


"""rebuild with chimaera, when it's working.

FOR NOW, loaded node paths will show backslashes instead of forward slashes.
Fixing this would require patching pathlib in a non-insane way,
which is beyond me
"""

@dataclass
class ButtonMode:
	"""button mode"""
	text:str
	target:T.Callable
	enabled: bool = True
	tooltip : str = ""


class ModeButton(QtWidgets.QWidget):
	"""button supporting different text, enabled state
	and target function depending on current mode.

	Probably overkill, for now just use separate buttons

	"""

class IoNodeWidget(QtWidgets.QWidget, BorderFrame, WpWidgetBase):
	"""represent a single scene io node -
	also allow altering its output path.

	No provision for changing input/output mode live -
	create a new object for that
	"""
	_baseCls = QtWidgets.QWidget

	def __init__(self, parent=None, mode=const.IoMode.Input):
		self._baseCls.__init__(self, parent=parent)
		WpWidgetBase.__init__(self)


		self.mode : const.IoMode.T() = mode
		self.node : WN = None
		self.tracker = NodeTracker(self)
		strParams = TreeFieldParams(
			showLabel=True,
			isPath=True,
			placeholderText=f"relative {self.mode.clsNameCamel()} path",

		)
		field = TreeField("ioPath", params=strParams)
		self.pathWidget = StringTreeWidget(field, parent=self)

		self.ioBtn = QtWidgets.QPushButton(f"Run {self.mode}", self)

		self.makeLayout()
		self.makeConnections()

	def paintEvent(self, arg__1:PySide2.QtGui.QPaintEvent) -> None:
		"""draw border"""
		#print("top paint event")
		self._baseCls.paintEvent(self, arg__1)
		BorderFrame.paintEvent(self, arg__1)

	def makeLayout(self):
		vl = QtWidgets.QVBoxLayout(self)
		hl = QtWidgets.QHBoxLayout()
		hl.addWidget(self.tracker)
		hl.addWidget(self.pathWidget)
		hl.setContentsMargins(0, 0, 0, 0)
		vl.addLayout(hl)
		vl.addWidget(self.ioBtn)
		vl.setContentsMargins(4, 4, 4, 4)
		self.setLayout(vl)

	def _onIoBtnPressed(self, *args, **kwargs):
		"""triggered when io button clicked"""
		if self.mode == const.IoMode.Input:
			lib.runInput()
		else:
			lib.runOutput(self.node)

	def makeConnections(self):
		self.ioBtn.pressed.connect(self._onIoBtnPressed)
		pass

	def _onTreePathChanged(self, *args, **kwargs):
		"""triggers whenever tree field value changes for any reason -
		save path change back to tree"""
		data = lib.nodeIoData(self.node)
		data.path = self.pathWidget.tree.value
		lib.setNodeIoData(self.node, data)
		self.node.saveAuxTree()

	def setNode(self, node:WN):
		"""set node to display
		node will already be set up as IO node with aux tree"""
		assert lib.isIoNode(node), "node must be io node"
		self.tracker.setNode(node)
		self.node = node

		self.palette().setColor(QtGui.QPalette.WindowText, QtGui.QColor(
			*self.mode.colour
		))
		#print("set node path", lib.nodeIoPath(node))
		self.pathWidget.tree.setValuePropagate(lib.nodeIoPath(node))
		self.connectToOwnedSlot(
			self.pathWidget.tree.getSignalComponent().valueChanged,
			self._onTreePathChanged)


class ChildListWidget(QtWidgets.QWidget, WpWidgetBase):
	"""widget holding list of child widgets, instead of items"""
	def __init__(self, parent=None):
		super(ChildListWidget, self).__init__(parent=parent)
		WpWidgetBase.__init__(self)

		vl = QtWidgets.QVBoxLayout(self)
		vl.setContentsMargins(0, 0, 0, 0)
		vl.setSpacing(0)
		self.setLayout(vl)

	def addWidget(self, widget:QtWidgets.QWidget):
		"""add widget to list"""
		widget.setParent(self)
		self.layout().addWidget(widget)

	def widgets(self):
		return [self.layout().itemAt(i).widget() for i in range(self.layout().count())]

	def clear(self):
		"""clear all widgets"""
		#print("clearing list")
		while self.layout().count():
			self.layout().takeAt(0).widget().deleteWp()
		self.adjustSize()

	def widgetList(self)->T.List[QtWidgets.QWidget]:
		"""return list of widgets"""
		return [self.layout().itemAt(i).widget() for i in range(self.layout().count())]

	def sizeHint(self) -> PySide2.QtCore.QSize:
		"""expand to all list items"""
		return self.minimumSizeHint()

	def minimumSizeHint(self) -> PySide2.QtCore.QSize:
		"""expand to all list items"""
		return self.layout().minimumSize()

def getSingleSel()->WN:
	return WN(cmds.ls(sl=True, l=True)[0])

class IoListPane(WpWidgetBase, QtWidgets.QGroupBox):
	"""holds list and any buttons for either input or output groups
	"""
	_baseCls = QtWidgets.QGroupBox

	def __init__(self, parent=None, mode:const.IoMode.T() = const.IoMode.Input):
		self._baseCls.__init__(self, parent=parent)
		WpWidgetBase.__init__(self)

		self.mode = mode
		self.newGrpBtn = QtWidgets.QPushButton(f"New {self.mode} Group", self)
		self.setGrpBtn = QtWidgets.QPushButton(f"Set as {self.mode} Group", self)
		self.removeGrpBtn = QtWidgets.QPushButton(f"Remove {self.mode} Group", self)
		self.refreshBtn = QtWidgets.QPushButton("Refresh", self)

		self.panelList = ChildListWidget(self)

		self._makeLayout()
		self._makeConnections()

		self.setTitle(f"{self.mode} Groups")
		self.setObjectName(f"{self.mode.clsNameCamel()}")
		#self.setToolTip(self.toolTip())

	def _makeLayout(self):
		autoLayout(self, recurse=False) # love this thing


	def _refreshList(self):
		"""refresh list of io nodes"""
		self.panelList.clear()
		nodes = lib.listExportNodes() if self.mode == const.IoMode.Output else lib.listImportNodes()
		for node in nodes:
			w = IoNodeWidget(self, mode=self.mode)
			w.setNode(node)
			self.panelList.addWidget(
				w
			)

	def _refreshButtons(self):
		"""update active and inactive buttons based on user selection"""

		sel = cmds.ls(sl=True, type="transform")
		if not len(sel) == 1:
			# back to normal state
			self.setGrpBtn.setEnabled(False)
			self.removeGrpBtn.setEnabled(False)
			return

		sel = WN(sel[0])

		if lib.isIoNode(sel, mode=self.mode):
			self.setGrpBtn.setEnabled(True)
			self.removeGrpBtn.setEnabled(True)
			return
		elif lib.isIoNode(sel): # io node but not this mode
			self.setGrpBtn.setEnabled(False)
			self.removeGrpBtn.setEnabled(False)
			return

		elif sel.nodeType() == "transform" and not sel.isShapeTransform():  # transform, not shape
			self.setGrpBtn.setEnabled(True)
			self.removeGrpBtn.setEnabled(False)
			return

	def syncFromScene(self, *args, **kwargs):
		"""sync list with scene"""
		self._refreshList()
		self._refreshButtons()

	def _onNewGrpBtnPressed(self, *args, **kwargs):
		"""create new group"""
		lib.createIoNode(self.mode)
		self.syncFromScene()

	def _onSetGrpBtnPressed(self, *args, **kwargs):
		lib.setIoNode(getSingleSel(), self.mode)
		self.syncFromScene()

	def _onRemoveGrpBtnPressed(self, *args, **kwargs):
		lib.removeIoNode(getSingleSel())
		self.syncFromScene()

	def onSceneSelectionChanged(self, *args, **kwargs):
		self._refreshButtons()


	def _makeConnections(self):
		self.newGrpBtn.clicked.connect(self._onNewGrpBtnPressed)
		self.setGrpBtn.clicked.connect(self._onSetGrpBtnPressed)
		self.removeGrpBtn.clicked.connect(self._onRemoveGrpBtnPressed)
		self.refreshBtn.clicked.connect(self.syncFromScene)


class IoManagerWidget(QtWidgets.QWidget, WpWidgetBase):
	"""widget to manage importing and exporting to consistent paths
	one list for input, one for output"""

	widgetName = "ioManagerWidget"

	def __init__(self, parent=None):
		super(IoManagerWidget, self).__init__(parent=parent)
		WpWidgetBase.__init__(self)

		self.inputWidget = IoListPane(self, const.IoMode.Input)
		self.openExplorerBtn = QtWidgets.QPushButton("Open Explorer", self)
		self.outputWidget = IoListPane(self, const.IoMode.Output)

		self.setObjectName(self.widgetName)

		self._makeLayout()
		self._makeConnections()

	def _makeLayout(self):
		autoLayout(self, recurse=False) # love this thing

	def _makeConnections(self):
		"""make signal connections
		TODO: update on node creation/deletion"""
		self.connectToOwnedSlot(
			getSceneGlobals().listener.selectionChanged,
			self._onSceneSelectionChanged
		)
		self.openExplorerBtn.clicked.connect(self._onOpenExplorerBtnPressed)

	def _onOpenExplorerBtnPressed(self, *args, **kwargs):
		"""open explorer to current output path"""
		path = lib.getScenePath()
		if path:
			os.startfile(path.parent)

	def _newIoGrp(self, mode:const.IoMode.T()):
		tf = createWN("transform", "newIO_GRP")
		lib.setIoNode(tf, "_out/newIO", mode)
		return tf


	def _onSceneSelectionChanged(self, *args, **kwargs):
		"""called when scene selection changes"""
		self.inputWidget.onSceneSelectionChanged()
		self.outputWidget.onSceneSelectionChanged()

	def syncFromScene(self):
		"""sync inputs from scene"""
		self.inputWidget.syncFromScene()
		self.outputWidget.syncFromScene()

	def closeEvent(self, event:PySide2.QtGui.QCloseEvent) -> None:
		#print("close event")
		super(IoManagerWidget, self).closeEvent(event)
		#del self

	def close(self) -> bool:
		"""called when widget is closed
		:return: True if closed, False if not
		"""
		print("closing io manager widget")
		self.deleteWp()

		return super(IoManagerWidget, self).close()

	# def deleteLater(self) -> None:
	# 	print("deleting io manager widget")
	# 	# find a better way to copy this logic across all widgets
	# 	# maybe call it from cleanup(), not other way round
	# 	for i in self.children():
	# 		delete.deleteObjectTree(i)
	# 	#delete.deleteObjectTree(self)
	# 	super(IoManagerWidget, self).deleteLater()


def showTool():
	widg = IoManagerWidget()
	widg.syncFromScene()
	return window.showToolWindow(widg, deleteExisting=True, floating=True)

