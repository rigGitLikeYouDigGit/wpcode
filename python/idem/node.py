

from __future__ import annotations

import pprint
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets

from wplib import log, Sentinel, TypeNamespace

from chimaera import ChimaeraNode
from chimaera.ui import NodeDelegate
from idem.dcc import DCC#, Maya
class IdemGraph(ChimaeraNode):

	def getAvailableNodesToCreate(self)->list[str]:
		"""return a list of node types that this node can support as
		children - by default allow all registered types
		TODO: update this as a combined class/instance method

		top-level idem graph must start out by defining DCC sessions
		to host those nodes - first thing we do is check for them
		"""
		allowed = []
		# add DCC session nodes
		for k, v in self.nodeTypeRegister.items():
			if issubclass(v, DCCSessionNode):
				allowed.append(k)
		#return list(self.nodeTypeRegister.keys())

		#TODO:
		#   now filter remaining DCC-relevant nodes to only those that have a session
		#   node in this graph

		#TODO:
		# maybe find some way to grey these nodes out, add an indication message somehow
		return allowed

	def sessionNodes(self)->list[DCCSessionNode]:
		return [i for i in self.branches if isinstance(i, DCCSessionNode)]

class DCCSessionNode(ChimaeraNode):
	"""DCC session nodes should show their own status,
	in the graph they can be dormant if their session isn't running

	CONSIDER : for now we just show the inputs/outputs of the scene in this top-level graph;
	could we try and show a "process" node for each one, that takes you to the DCC
	chimaeara graph when you open it?

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


