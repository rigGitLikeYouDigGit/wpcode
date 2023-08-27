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


class NodeFnSet:
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

	FnSets are INSTANTIATED on chimaera nodes, and an attached function set's compute
	takes precedence over the node's value expression. Constructs may be destroyed and
	reinstantiated for each node - reloading a construct class should immediately update
	a node's active behaviour.

	"""

	def __init__(self, node:ChimaeraNode=None):
		self.node : ChimaeraNode = None
		if node is not None:
			self.setNode(node)

	def setNode(self, node:ChimaeraNode)->None:
		self.node = node
		if node is None:
			return
		node.setFnSet(self)

		# delegate methods from node
		self.name = node.name
		self.refMap = node.refMap
		self.resultParams = node.resultParams
		self.resultStorage = node.resultStorage


	@classmethod
	def setupNode(cls, node:ChimaeraNode, name:str, parent:ChimaeraNode=None)->None:
		"""default process to set up node when freshly created -
		used by plug nodes to create plugs, etc
		"""
		# set param defaults
		rawParams = node.sourceParams()
		rawParams.lookupCreate = True
		cls.makeDefaultParams(rawParams)
		rawParams.lookupCreate = False




	@classmethod
	def makeDefaultParams(cls, paramRoot: Tree):
		"""set up default resultParams for placing joints
		maybe pass in tree root as argument"""



	def compute(self, **kwargs)->object:
		"""compute node output from input data block
		return the node's new value to be cached"""
		raise NotImplementedError

	@classmethod
	def dirtyComputeOuter(cls, node:ChimaeraNode, graph:ChimaeraNode)->object:
		"""outer function called to run dirty compute
		return the node's new value to be cached
		should call compute and handle any possible errors
		"""
		raise NotImplementedError

	@classmethod
	def create(cls, name:str, parent:ChimaeraNode=None)->cls:
		"""create a new node with this construct -
		control flow is wacky, since this calls graph which will also call this
		"""
		node = parent.createNode(name, cls)
		return cls(node)