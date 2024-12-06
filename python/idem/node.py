

from __future__ import annotations

import pprint
import typing as T

from PySide2 import QtCore, QtGui, QtWidgets

from wplib import log, Sentinel, TypeNamespace
from wptree import Tree
from wpui.widget import LogWidget
from chimaera import ChimaeraNode
from chimaera.ui import NodeDelegate
from idem.dcc import DCC#, Maya

# TODO: ASSETS
#   how can an idem/chimaera graph know about things like current asset, current file path?
#   some kind of context object? assigned to the graph when the higher model changes?

if T.TYPE_CHECKING:
	from idem.model import IdemSession

class IdemGraph(ChimaeraNode):

	session : IdemSession

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

	session nodes don't have strong links to others, just collect separate IO nodes for their
	session in groups

	"""
	#if T.TYPE_CHECKING:
	parent : IdemGraph

	def __init__(self, data:Tree):
		super().__init__(data)
		# save live reference to this node's program process -
		# DO NOT serialise this
		self.dcc = self.dccType()(self.name) #initialised empty
		self.process = None

	@classmethod
	def dccType(cls)->type[DCC]:
		raise NotImplementedError
	@classmethod
	def prefix(cls) ->tuple[str]:
		return ("i", )
	@classmethod
	def canBeCreated(cls):
		return cls.__name__ != "DCCSessionNode"

	def createDCCProcess(self):
		"""use the name and type of this node to spawn a new process
		of the given DCC"""
		asset = self.parent.session.asset()
		log("create session of", self.dcc, "in asset", asset)
		self.dcc.launch(
			idemParams={"processName" : self.name,
			            "launchInDir" : str(self.parent.session.asset().diskPath()),
			            "portNumber" : 121312}
		)

	def getContextMenuTree(self,
	                       event:QtGui.QMouseEvent=None,
	                       uiData:dict=None) ->T.Optional[Tree]:
		t = Tree(self.name)
		if self.process is None:
			t["launch"] = self.createDCCProcess
		return t


"""
sending : blue
receiving : orange
status : 
	green for waiting on commands
	orange for working
	red for crash / error
	grey for inactive
"""


if __name__ == '__main__':

	app = QtWidgets.QApplication()
	w = QtWidgets.QGraphicsView()
	s = QtWidgets.QGraphicsScene(w)
	w.setScene(s)
	s.addPixmap(QtGui.QIcon(str(Maya.iconPath())).pixmap(30, 30))
	w.show()
	app.exec_()


