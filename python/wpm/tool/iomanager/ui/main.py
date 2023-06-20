from __future__ import annotations
import typing as T


from pathlib import Path

from PySide2 import QtWidgets, QtCore, QtGui

from wpm import cmds, om, oma, WN, createWN, getSceneGlobals

from .. import lib

"""rebuild with chimaera, when it's working"""

class IoNodeWidget(QtWidgets.QWidget):
	"""represent a single scene io node"""

	def __init__(self, parent=None):
		super(IoNodeWidget, self).__init__(parent=parent)

		self.node : WN = None
		self.label = QtWidgets.QLabel(self)

	def setNode(self, node:WN):
		"""set node to display"""
		assert lib.isIoNode(node), "node must be io node"
		self.node = node
		self.label.setText(node.name())
		self.palette().setColor(QtGui.QPalette.WindowText, QtGui.QColor(
			*lib.IoMode[node.getAuxTree()[lib.MODE_KEY]].colour
		))

class ChildListWidget(QtWidgets.QWidget):
	"""widget holding list of child widgets, instead of items"""
	def __init__(self, parent=None):
		super(ChildListWidget, self).__init__(parent=parent)
		self.layout = QtWidgets.QVBoxLayout(self)
		self.layout.setContentsMargins(0, 0, 0, 0)
		self.layout.setSpacing(0)

	def addWidget(self, widget:QtWidgets.QWidget):
		"""add widget to list"""
		widget.setParent(self)
		self.layout.addWidget(widget)

	def clear(self):
		while self.layout.count():
			self.layout.takeAt(0).widget().deleteLater()

	def widgetList(self)->T.List[QtWidgets.QWidget]:
		"""return list of widgets"""
		return [self.layout.itemAt(i).widget() for i in range(self.layout.count())]


class IoManagerWidget(QtWidgets.QWidget):
	"""widget to manage importing and exporting to consistent paths
	one list for input, one for output"""
	def __init__(self, parent=None):
		super(IoManagerWidget, self).__init__(parent=parent)
		# self.syncInputsBtn = QtWidgets.QPushButton("Sync Inputs", self)
		# self.inputList = ChildListWidget(self)

		self.makeIoGrpBtn = QtWidgets.QPushButton("Make new IO Group", self)
		self.removeIoGrpBtn = QtWidgets.QPushButton("Remove IO Group", self)

		self.outputsLabel = QtWidgets.QLabel("Outputs", self)
		self.refreshOutputsBtn = QtWidgets.QPushButton("Refresh Outputs", self)
		self.outputList = ChildListWidget(self)
		self.exportBtn = QtWidgets.QPushButton("Export Outputs", self)

		self.makeLayout()

	def makeLayout(self):
		self.setLayout(QtWidgets.QVBoxLayout(self))
		hl = QtWidgets.QHBoxLayout(self)
		hl.addWidget(self.outputsLabel)
		hl.addWidget(self.refreshOutputsBtn)
		self.layout().addLayout(hl)
		self.layout().addWidget(self.outputList)
		self.layout().addWidget(self.exportBtn)

	def makeConnections(self):
		"""make signal connections"""
		getSceneGlobals().listener.selectionChanged.connect(self._onSceneSelectionChanged)

	def _refreshButtonLabels(self):
		"""refresh button labels
		if single transform is selected, allow conversion to io group
		else only allow creating new
		"""
		sel = cmds.ls(sl=True, type="transform")
		if len(sel) == 1:
			sel = WN(sel[0])

			if lib.isIoNode(sel):
				self.makeIoGrpBtn.setText("IO Group selected")
				self.makeIoGrpBtn.setEnabled(False)
				self.removeIoGrpBtn.setEnabled(True)
			else:
				self.makeIoGrpBtn.setText("Promote selected to IO Group")
				self.makeIoGrpBtn.setEnabled(True)
				self.removeIoGrpBtn.setEnabled(False)
		else:
			self.makeIoGrpBtn.setText("Make new IO Group")
			self.removeIoGrpBtn.setEnabled(False)


	def _onSceneSelectionChanged(self, *args, **kwargs):
		"""called when scene selection changes"""
		self.refreshOutputs()



	def refreshOutputs(self):
		"""refresh output list"""
		self.outputList.clear()
		for node in lib.listExportNodes():
			widget = IoNodeWidget(self)
			widget.setNode(node)
			self.outputList.addWidget(widget)