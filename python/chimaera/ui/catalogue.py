
from __future__ import annotations

import pprint
import typing as T


from PySide2 import QtCore, QtWidgets, QtGui

from wplib import log, Sentinel, TypeNamespace

from wpui.widget.canvas import *


from chimaera import ChimaeraNode

"""
widget to show when pressing tab, showing available nodes
for this graph
"""

class NodeCatalogue(QtWidgets.QLineEdit):
	"""

	TODO: this is very simple for now
	 - improve the text search bar
	 - allow thumbnails and visual palette for nodes representing
	    tools, assets, etc
	"""

	nodeSelected = QtCore.Signal(str)

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setPlaceholderText("Create node...")
		self.setCompleter(QtWidgets.QCompleter(parent=self))
		self.availableNodeKeys = []

		self.textEdited.connect(self.onTextEdited)

	def setAvailableNodes(self, nodes:list[str]):
		self.availableNodeKeys = nodes
		self.completer().setModel(QtCore.QStringListModel(nodes, parent=None))

	def onTextEdited(self, s:str):
		"""check that requested node is valid -
		then if so, pass it on to the signal, then close this window"""
		if not s in self.availableNodeKeys:
			log("invalid node type requested", s, "aborting")
			self.close()
			return
		self.close()
		self.nodeSelected.emit(s)
		return




