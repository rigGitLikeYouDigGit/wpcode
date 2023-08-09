from __future__ import annotations
import typing as T

from wplib import Expression
from wptree import Tree

from chimaera.core.node import ChimaeraNode
if T.TYPE_CHECKING:
	from chimaera.core.graph import ChimaeraGraph

"""not sure if we should have a type disconnect between base nodes
and plug nodes - we could probably combine them if needed.


Plug node is associated with separate Chimaera nodes forming input
and output plug trees, managed by main node

Unsure if these should automatically be subgraphs or live at same
hierarchy level as main node.

PlugNode classes should REALLY be function owners operating on a 
datablock, in the same model as Maya -
separates operation and data, makes it easier to serialise and merge data,
makes it more robust to missing class references, etc



refer to node PLUGS as TREES
PLUGS are NODES
a plug has its own ref map, pointing to child plugs,
has reserved ref "_src" to the driver(s) of the plug

default value operation of plug node is to combine all child plugs
and input into single data object

when iterating over plug nodes as tree:
tree NAME is name of plug
tree VALUE is plug data object
actually no, just make a shim around actual nodes, name and value should line up
perfectly, and can access refMap and other node attributes directly
"""





class PlugNode(ChimaeraNode):
	"""Plug node is really just a few ChimaeraNodes in a trenchcoat

	PLUG: chimaera node acting as single socket interface to graph
	PLUG NODE CENTRE: chimaera node that owns plug nodes
	"""
	IN_PLUG_KEY = "_in"
	OUT_PLUG_KEY = "_out"
	PLUG_SOURCE_KEY = "_src"

	# region plug node support
	@classmethod
	def setPlugSource(cls, plugNode:ChimaeraNode, plugDriver:ChimaeraNode)->None:
		"""set plug node to be driven by another node's plug"""
		plugNode.setRef(cls.PLUG_SOURCE_KEY, uid=plugDriver.uid)

	@classmethod
	def plugSource(cls, plugNode:ChimaeraNode, raw=False)->list[ChimaeraNode]:
		"""get source node of plug node"""
		sourceNodeFilter = plugNode.getRef(cls.PLUG_SOURCE_KEY, raw=False)
		if raw:
			return sourceNodeFilter
		return plugNode.graph().resolveRef(sourceNodeFilter, fromNode=plugNode)

	@classmethod
	def nodeIsPlug(cls, node:ChimaeraNode):
		"""if a node defines _src, it's a plug of another node
		"""
		return cls.PLUG_SOURCE_KEY in node.refMapRaw()

	@classmethod
	def plugChildren(cls, plug:ChimaeraNode)->dict[str, list[ChimaeraNode]]:
		"""get immediate children of plug node"""
		fullMap = plug.graph().resolveRefMap(plug.refMap(), plug)
		fullMap.pop(cls.PLUG_SOURCE_KEY, None)
		return fullMap

	#endregion

	@classmethod
	def makeRootPlugNodes(cls, node:ChimaeraNode, graph:ChimaeraGraph)->None:
		"""create root plug nodes for this node and link to refmap
		"""
		inRoot = graph.createNode("_in", ChimaeraNode)
		node.setRef(cls.IN_PLUG_KEY, uid=inRoot.uid)
		#set value input of plug node to be empty
		cls.setPlugSource(inRoot, None)



		# outRoot = graph.createNode("_out", ChimaeraNode)
		# node.setRef("_out", uid=outRoot.uid)



	@classmethod
	def makePlugs(cls, inRoot:Tree, outRoot:Tree)->None:
		"""create plugs for this node - work with trees, not plug nodes.
		"""

	@classmethod
	def syncInputPlugs(cls, fromInTree:Tree)->None:
		"""regenerate input plug nodes from tree -
		to be called if plugs change during compute
		"""

	@classmethod
	def setupNode(cls, node:ChimaeraNode, graph:ChimaeraGraph) ->None:
		"""control flow here is wacky but it's just a test -
		create trees of plug nodes, then set up the node
		"""
	@classmethod
	def defaultParams(cls, paramRoot:Tree)->Tree:
		"""set up default params for placing joints
		maybe pass in tree root as argument"""
		return paramRoot
