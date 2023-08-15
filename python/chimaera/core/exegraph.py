

from __future__ import annotations
import typing as T

from wplib import Expression, DirtyExp, coderef
from wplib.object import DirtyNode, DirtyGraph
from wplib.sentinel import Sentinel
from wplib.object import UidElement

from wptree import Tree

if T.TYPE_CHECKING:
	from chimaera.core.node import ChimaeraNode

"""We consider the data and execution of Chimaera separately - 
main nodes hold data and subnetworks, while this class handles
dirty propagation and execution.

Seems necessary that execution have a global view of the graph - 
therefore single object handling arbitrary chimaera islands.

"""


class ChimaeraDirtyGraph(DirtyGraph):

	def addChimaeraNode(self, node: ChimaeraNode):
		"""add a node to the graph"""
		nodes = {node}.union( set(node.allChildNodes().values()) )
		self.addNodesAndPrecedents(nodes)






