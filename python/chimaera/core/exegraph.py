

from __future__ import annotations
import typing as T

from wplib import Expression, DirtyExp, CodeRef
from wplib.object import DirtyNode, DirtyGraph
from wplib.sentinel import Sentinel
from wplib.object import UidElement

from wptree import Tree

from chimaera.core.node import ChimaeraNode
if T.TYPE_CHECKING:
	pass

"""We consider the data and execution of Chimaera separately - 
main nodes hold data and subnetworks, while this class handles
dirty propagation and execution.

Seems necessary that execution have a global view of the graph - 
therefore single object handling arbitrary chimaera islands.

"""

class ChimaeraDirtyGraph(DirtyGraph):

	def __init__(self):
		"""init dirty graph"""
		super().__init__()
		self._nodeRegisterDNode : DirtyNode = None

	def getNodeRegister(self):
		return {"uid" : self.allChimaeraNodes(),
		        "name" : {n.name() : n for n in self.allChimaeraNodes().values()}
		        }

	@classmethod
	def makeRegisterNode(cls, graph:ChimaeraDirtyGraph) ->DirtyNode:
		"""make a node to register names of all nodes in the graph,
		marked dirty when a name changes.

		This is my first attempt at making a centralised nodes to support
		list or filter operations - pretty sure it's a stupid idea since
		we immediately hit cycles.

		I think we might just have to live with cycles.


		"""
		node = DirtyNode("nameRegister")
		node.dirtyComputeFn = graph.getNodeRegister
		graph.add_node(node)
		graph._nodeRegisterDNode = node
		return node


	@classmethod
	def create(cls):
		"""make new dirty graph, add global node for name register"""
		graph = cls()
		registerNode = graph.makeRegisterNode(graph)
		return graph

	def allChimaeraNodes(self)->dict[str, ChimaeraNode]:
		"""get all nodes in the graph"""
		return {n.uid : n for n in self.nodes if isinstance(n, ChimaeraNode)}

	def postAddChimaeraNode(self, node: ChimaeraNode):
		"""run additional connections after main chimaera structure added to
		graph"""
		self.add_edge(node.nameExp(), self._nodeRegisterDNode)

	def addChimaeraNode(self, node: ChimaeraNode):
		"""add a node to the graph"""
		nodes = {node}.union( set(node.allChildNodes().values()) )
		self.addNodesAndPrecedents(nodes)
		for i in nodes:
			self.postAddChimaeraNode(i)
		self.setNodeDirty(node)






