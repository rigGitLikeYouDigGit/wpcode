from __future__ import annotations
import typing as T


from pathlib import Path

from PySide2 import QtWidgets, QtCore, QtGui

from dataclasses import dataclass

from tree.lib.object import TypeNamespace
from tree import Tree

from wp.treefield import TreeField
from wp.ui.layout import autoLayout
from wp.ui.treefieldwidget import TreeFieldParams, StringWidget
from wpm import cmds, om, oma, WN, createWN, getSceneGlobals

from wpm.lib.ui import window
from wpm.lib.ui.nodetracker import NodeTracker
from .. import lib


"""rebuild with chimaera, when it's working"""

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


class IoNodeWidget(QtWidgets.QWidget):
	"""represent a single scene io node -
	also allow altering its output path"""

	def __init__(self, parent=None):
		super(IoNodeWidget, self).__init__(parent=parent)

		self.node : WN = None
		self.tracker = NodeTracker(self)
		strParams = TreeFieldParams(
			showLabel=True,
			isPath=True,
			placeholderText="relative io path"
		)

		self.pathWidget = StringWidget(TreeField("ioPath"), parent=self)
		self.ioBtn = QtWidgets.QPushButton("Run IO", self)

		self._signalsToClean = []

		self.makeLayout()
		self.makeConnections()

	def makeLayout(self):
		vl = QtWidgets.QVBoxLayout(self)
		hl = QtWidgets.QHBoxLayout()
		hl.addWidget(self.tracker)
		hl.addWidget(self.pathWidget)
		vl.addLayout(hl)
		vl.addWidget(self.ioBtn)

	def makeConnections(self):
		pass

	def setNode(self, node:WN):
		"""set node to display
		node will already be set up as IO node with aux tree"""
		assert lib.isIoNode(node), "node must be io node"
		self.tracker.setNode(node)
		self.node = node
		#self.tracker.setText(node.name())
		self.palette().setColor(QtGui.QPalette.WindowText, QtGui.QColor(
			*lib.IoMode[node.getAuxTree()[lib.MODE_KEY]].colour
		))
		self.pathWidget.tree.setValuePropagate(lib.nodeIoPath(node))

class ChildListWidget(QtWidgets.QWidget):
	"""widget holding list of child widgets, instead of items"""
	def __init__(self, parent=None):
		super(ChildListWidget, self).__init__(parent=parent)
		vl = QtWidgets.QVBoxLayout(self)
		vl.setContentsMargins(0, 0, 0, 0)
		vl.setSpacing(0)
		self.setLayout(vl)

	def addWidget(self, widget:QtWidgets.QWidget):
		"""add widget to list"""
		widget.setParent(self)
		self.layout().addWidget(widget)

	def clear(self):
		while self.layout().count():
			self.layout().takeAt(0).widget().deleteLater()

	def widgetList(self)->T.List[QtWidgets.QWidget]:
		"""return list of widgets"""
		return [self.layout().itemAt(i).widget() for i in range(self.layout().count())]


class IoManagerWidget(QtWidgets.QWidget):
	"""widget to manage importing and exporting to consistent paths
	one list for input, one for output"""

	widgetName = "ioManagerWidget"

	def __init__(self, parent=None):
		super(IoManagerWidget, self).__init__(parent=parent)
		# self.syncInputsBtn = QtWidgets.QPushButton("Sync Inputs", self)
		# self.inputList = ChildListWidget(self)

		self._signalsToClean = []


		self.makeOutputGrpBtn = QtWidgets.QPushButton("Make new Output Group", self)
		self.resetAsOutputGrpBtn = QtWidgets.QPushButton("Reset as Output Group", self)
		self.removeIoGrpBtn = QtWidgets.QPushButton("Remove IO Group", self)

		self.outputsLabel = QtWidgets.QLabel("Outputs", self)
		self.refreshOutputsBtn = QtWidgets.QPushButton("Refresh Outputs", self)
		self.outputList = ChildListWidget(self)
		self.exportBtn = QtWidgets.QPushButton("Export Outputs", self)

		self.setObjectName(self.widgetName)

		self._makeLayout()
		self._makeConnections()

	def _makeLayout(self):
		autoLayout(self, recurse=False) # love this thing

	def _makeConnections(self):
		"""make signal connections"""
		getSceneGlobals().listener.selectionChanged.connect(self._onSceneSelectionChanged)
		self._signalsToClean.append((getSceneGlobals().listener.selectionChanged, self._onSceneSelectionChanged))

		self.makeOutputGrpBtn.clicked.connect(self._onMakeOutputGrpBtnPressed)
		self.resetAsOutputGrpBtn.clicked.connect(self._onResetAsOutputGrpBtnPressed)
		self.removeIoGrpBtn.clicked.connect(self._onRemoveIoGrpBtnPressed)

	def _newIoGrp(self, mode:lib.IoMode.T()):
		tf = createWN("transform", "newIO_GRP")
		lib.setIoNode(tf, "_out/newIO", mode)
		return tf

	def _onMakeOutputGrpBtnPressed(self, *args, **kwargs):
		"""create new transform, set it up as io group"""
		self._newIoGrp(lib.IoMode.Export)
		self.syncFromScene()

	def _onResetAsOutputGrpBtnPressed(self, *args, **kwargs):
		"""reset selected transform as output group"""
		sel = cmds.ls(sl=True, type="transform")
		assert len(sel) == 1, "select one transform"
		sel = WN(sel[0])
		lib.setIoNode(sel, "_out/" + sel.name(), lib.IoMode.Export)
		self.syncFromScene()

	def _onRemoveIoGrpBtnPressed(self, *args, **kwargs):
		"""remove io group"""
		sel = cmds.ls(sl=True, type="transform")
		assert len(sel) == 1, "select one transform"
		sel = WN(sel[0])
		lib.removeIoNode(sel)
		self.syncFromScene()

	def _refreshButtonLabels(self):
		"""refresh button labels
		if single transform is selected, allow conversion to io group
		else only allow creating new
		"""
		sel = cmds.ls(sl=True, type="transform")
		if not len(sel) == 1:
			# back to normal state
			self.resetAsOutputGrpBtn.setEnabled(False)
			self.removeIoGrpBtn.setEnabled(False)
			return

		sel = WN(sel[0])
		print("is shape tf", sel.isShapeTransform(), sel.nodeType())

		if lib.isIoNode(sel):
			self.resetAsOutputGrpBtn.setEnabled(True)
			self.removeIoGrpBtn.setEnabled(True)
			return
		elif sel.nodeType() == "transform" and not sel.isShapeTransform(): # transform, not shape
			self.resetAsOutputGrpBtn.setEnabled(True)
			self.removeIoGrpBtn.setEnabled(False)
			return


	def _onSceneSelectionChanged(self, *args, **kwargs):
		"""called when scene selection changes"""
		self._refreshOutputList()
		self._refreshButtonLabels()


	def _refreshOutputList(self):
		"""refresh output list"""
		self.outputList.clear()
		for node in lib.listExportNodes():
			widget = IoNodeWidget(self.outputList)
			self.outputList.addWidget(widget)
			widget.setNode(node)

	def syncFromScene(self):
		"""sync inputs from scene"""
		self._refreshOutputList()
		self._refreshButtonLabels()

	def closeEvent(self, event:PySide2.QtGui.QCloseEvent) -> None:
		print("close event")
		return super(IoManagerWidget, self).closeEvent(event)

	def close(self) -> bool:
		"""called when widget is closed
		:return: True if closed, False if not
		"""
		print("closing io manager widget")
		for i in self._signalsToClean:
			i[0].disconnect(i[1])
		return super(IoManagerWidget, self).close()



def showTool():
	widg = IoManagerWidget()
	widg._onSceneSelectionChanged()
	return window.showToolWindow(widg, deleteExisting=True, floating=True)

