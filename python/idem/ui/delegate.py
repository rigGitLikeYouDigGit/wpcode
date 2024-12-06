from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtGui, QtWidgets
from wpui.widget import LogWidget
from chimaera.ui import NodeDelegate
from idem.node import DCCSessionNode
from idem.ui.processwidget import ProcessStatusWidget


class SessionNodeDelegate(NodeDelegate):
	"""
	- show DCC icon
	- show log window for connected process
	- show status lights for session next to
	"""
	forTypes = (DCCSessionNode, )

	node : DCCSessionNode

	def __init__(self, node, parent=None):
		super().__init__(node, parent)

		self.logW = LogWidget(parent=self.w)
		self.wLayout.addWidget(self.logW)

		self.processW = ProcessStatusWidget(parent=self.w)
		self.wLayout.insertWidget(1, self.processW)

	def icon(self) -> (None, QtGui.QIcon):
		"""return icon to show for node - by default nothing"""
		return QtGui.QIcon(str(self.node.dcc.iconPath())
		                   )
