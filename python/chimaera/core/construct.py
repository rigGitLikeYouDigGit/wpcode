from __future__ import annotations
import typing as T

from wplib import Expression
from wptree import Tree

if T.TYPE_CHECKING:
	#from chimaera.core.graph import ChimaeraGraph
	from chimaera.core.node import ChimaeraNode

	pass

"""
Construct wrapper base for operating on chimaera nodes at high level
"""


class NodeConstruct:
	"""it lives again lol
	Base class for "function set" wrapper objects to be created
	around base chimaera nodes -
	taking inspiration from how Maya handles MObjects and MFn function sets.
	Subclassing ChimaeraNode shouldn't ever really be necessary-
	function sets should define "class" of node.

	For now, tie each chimaera node to the class of construct that
	created it, but in future, could definitely do a type-like system
	of parent-class constructs operating on nodes created by child-class
	constructs.

	Main point is that a chimaera node may outlive or predate its creator construct,

	"""

	def __init__(self, node:ChimaeraNode=None):
		self.node : ChimaeraNode = node

	@classmethod
	def setupNode(cls, node:ChimaeraNode, parent:ChimaeraNode=None)->None:
		"""default process to set up node when freshly created -
		used by plug nodes to create plugs, etc
		"""

	@classmethod
	def compute(cls, dataBlock:dict[str, T.Any], graph:ChimaeraNode)->None:
		"""compute node output from input data block"""
		raise NotImplementedError