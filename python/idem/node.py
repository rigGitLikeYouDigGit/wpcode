

from __future__ import annotations

import pprint
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets

from wplib import log, Sentinel, TypeNamespace

from chimaera import ChimaeraNode
from chimaera.ui import NodeDelegate
from idem.dcc import DCC, Maya
class IdemGraph(ChimaeraNode):

	def getAvailableNodesToCreate(self)->list[str]:
		"""return a list of node types that this node can support as
		children - by default allow all registered types
		TODO: update this as a combined class/instance method
		"""
		return list(self.nodeTypeRegister.keys())

class DCCSessionNode(ChimaeraNode):
	"""DCC session nodes should show their own status,
	in the graph they can be dormant if their session isn't running
	"""
	@classmethod
	def dcc(cls)->type[DCC]:
		raise NotImplementedError
	@classmethod
	def prefix(cls) ->tuple[str]:
		return ("i", )
	@classmethod
	def canBeCreated(cls):
		return cls.__name__ != "DCCSessionNode"

class MayaSessionNode(DCCSessionNode):
	@classmethod
	def dcc(cls)->type[Maya]:
		return Maya

"""
sending : blue
receiving : orange
status : 
	green for waiting on commands
	orange for working
	red for crash / error
	grey for inactive
"""

class SessionNodeDelegate(NodeDelegate):
	forTypes = (DCCSessionNode, )

	node : DCCSessionNode

	def icon(self) -> (None, QtGui.QIcon):
		"""return icon to show for node - by default nothing"""
		return QtGui.QIcon(str(self.node.dcc().iconPath())
		                   )


if __name__ == '__main__':

	app = QtWidgets.QApplication()
	w = QtWidgets.QGraphicsView()
	s = QtWidgets.QGraphicsScene(w)
	w.setScene(s)
	s.addPixmap(QtGui.QIcon(str(Maya.iconPath())).pixmap(30, 30))
	w.show()
	app.exec_()


