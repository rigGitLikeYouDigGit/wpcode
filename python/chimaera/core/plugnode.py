from __future__ import annotations
import typing as T

from wplib.sequence import getFirst, flatten
from wplib import Expression
from wptree import Tree

from chimaera.core.node import ChimaeraNode, RefValue, newRefValue
from chimaera.core.construct import NodeFnSet
if T.TYPE_CHECKING:
	#from chimaera.core.graph import ChimaeraGraph
	pass

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



class PlugNode(NodeFnSet):
	"""Plug node is really just a few ChimaeraNodes in a trenchcoat


	PLUG: chimaera node acting as single socket interface to graph
	PLUG NODE CENTRE: chimaera node that owns plug nodes
	"""
	IN_PLUG_KEY = "_in"
	OUT_PLUG_KEY = "_out"
	PLUG_SOURCE_KEY = "_src"
	PLUG_PARENT_KEY = "_parent"
	PLUG_CHILDREN_KEY = "_children"
	PLUG_MAIN_KEY = "_plugMain"

	# region plug node support
	@classmethod
	def setPlugSource(cls, plugNode:ChimaeraNode, plugDriver:ChimaeraNode)->None:
		"""set plug node to be driven by another node's plug"""
		if isinstance(plugDriver, ChimaeraNode):
			plugNode.setRef(cls.PLUG_SOURCE_KEY, uid=(plugDriver.uid,))
		elif plugDriver is None:
			plugNode.setRef(cls.PLUG_SOURCE_KEY, uid=())

	@classmethod
	def plugSourceRefValue(cls, plugNode:ChimaeraNode)->T.Any:
		"""get value of plug source ref"""
		return plugNode.getRef(cls.PLUG_SOURCE_KEY, )

	@classmethod
	def getPlugParent(cls, childPlug:ChimaeraNode)->ChimaeraNode:
		"""get parent plug of child plug"""
		parentPlug = getFirst(childPlug.parent().resolveChildRef(
			childPlug.getRef(cls.PLUG_PARENT_KEY, raw=False),
			fromNode=childPlug))
		return parentPlug

	@classmethod
	def setPlugParent(cls, childPlug:ChimaeraNode, parentPlug:ChimaeraNode)->None:
		"""set parent plug of child plug
		feels very close to tree, but we're not quite ready for that yet"""
		if isinstance(parentPlug, ChimaeraNode):
			childPlug.setRef(cls.PLUG_PARENT_KEY, uid=(parentPlug.uid,))

		elif parentPlug is None:
			parent = cls.getPlugParent(childPlug)
			if parent:
				# remove parent
				parentRef : RefValue = parent.getRef(cls.PLUG_CHILDREN_KEY, raw=True)
				parentRef["uid"].remove(childPlug.uid)
				parent.setRef(cls.PLUG_CHILDREN_KEY, refVal=parentRef)
			childPlug.setRef(cls.PLUG_PARENT_KEY, uid=())

	@classmethod
	def plugSource(cls, plugNode:ChimaeraNode, raw=False)->tuple[ChimaeraNode]:
		"""get source node of plug node"""
		# sourceNodeFilter = plugNode.getRef(cls.PLUG_SOURCE_KEY, raw=False)
		# # if raw:
		# # 	return sourceNodeFilter
		refVal = plugNode.getRef(cls.PLUG_SOURCE_KEY,)
		print("ref val", refVal)
		if refVal is None:
			return ()
		return plugNode.parent().resolveChildRef(refVal, plugNode)


	@classmethod
	def nodeIsPlug(cls, node:ChimaeraNode):
		"""if a node defines _src, it's a plug of another node
		"""
		return cls.PLUG_MAIN_KEY in node.refMapRaw()

	@classmethod
	def plugMainNode(cls, plugNode:ChimaeraNode)->ChimaeraNode:
		"""get main node of plug node"""
		return plugNode.parent().resolveChildRef(plugNode.getRef(cls.PLUG_MAIN_KEY, ), plugNode)[0]

	@classmethod
	def plugIsInput(cls, plug:ChimaeraNode)->bool:
		"""return true if plug is root input,
		or child of root input
		"""
		# return plug.name() == cls.IN_PLUG_KEY

	# region plug traversal
	# yes I know we need proper tree support here
	@classmethod
	def plugChildren(cls, plug:ChimaeraNode)->tuple[ChimaeraNode]:
		"""get immediate children of plug node"""
		nodeSet = plug.parent().resolveChildRef(plug.getRef(cls.PLUG_CHILDREN_KEY, default=newRefValue()), plug)
		return nodeSet

	@classmethod
	def plugChildMap(cls, plug:ChimaeraNode)->dict[str, ChimaeraNode]:
		"""get child map of plug node"""
		childMap = {}
		for child in cls.plugChildren(plug):
			childMap[child.name()] = child
		return childMap

	#endregion

	@classmethod
	def inputPlugRoot(cls, mainNode:ChimaeraNode):
		allNodes = mainNode.resolveChildRef(mainNode.getRef(cls.IN_PLUG_KEY, ), mainNode)
		return getFirst(allNodes)

	def ownInputPlugRoot(self)->ChimaeraNode:
		return self.inputPlugRoot(self.node)

	@classmethod
	def outputPlugRoot(cls, mainNode:ChimaeraNode):
		allNodes = mainNode.resolveChildRef(mainNode.getRef(cls.OUT_PLUG_KEY, ), mainNode)
		return getFirst(allNodes)

	def ownOutputPlugRoot(self)->ChimaeraNode:
		return self.outputPlugRoot(self.node)

	@classmethod
	def _makeRootPlugNodes(cls, mainNode:ChimaeraNode, graph:ChimaeraNode)->tuple[ChimaeraNode, ChimaeraNode]:
		"""create root plug nodes for this node and link to refmap
		"""
		inRoot = graph.createNode(cls.IN_PLUG_KEY)
		mainNode.setRef(cls.IN_PLUG_KEY, uid=(inRoot.uid,))
		#set value input of plug node to be empty
		cls.setPlugSource(inRoot, None)

		outRoot = graph.createNode(cls.OUT_PLUG_KEY)
		mainNode.setRef(cls.OUT_PLUG_KEY, uid=(outRoot.uid,), affectEval=False)
		cls.setPlugSource(outRoot, mainNode)
		return inRoot, outRoot


	@classmethod
	def makePlugs(cls, inRoot:Tree, outRoot:Tree)->None:
		"""OVERRIDE :
		create default plugs for this node - work with trees, not plug nodes.
		"""

	@classmethod
	def syncInputPlugs(cls, fromInTree:Tree)->None:
		"""regenerate input plug nodes from tree -
		to be called if plugs change during compute
		"""

	@classmethod
	def _makePlugNodesFromTree(cls, rootNode:ChimaeraNode, plugTree:Tree, mainNode:ChimaeraNode):
		for i in plugTree.allBranches(includeSelf=True, depthFirst=False):
			if not i.parent:
				i.node = rootNode
				continue
			#make plug node
			plugNode = rootNode.parent().createNode(i.name)
			i.node = plugNode
			#set plug node parent
			plugNode.setRef(cls.PLUG_PARENT_KEY, uid=i.parent.node.uid)
			plugNode.setRef(cls.PLUG_MAIN_KEY, uid=(mainNode.uid,))


	#endregion

	@classmethod
	def extractInputData(cls, node:ChimaeraNode)->Tree:
		"""return data tree from node's input plugs"""

	@classmethod
	def setupNode(cls, node:ChimaeraNode, name:str, parent:ChimaeraNode=None) ->None:
		"""control flow here is wacky but it's just a test -
		create trees of plug nodes, then set up the node

		make plug setup explicit here, reduce nesting
		"""
		super().setupNode(node, name, parent)
		#make root plug nodes
		#cls._setupAllPlugs(node, parent)
		inRoot, outRoot = cls._makeRootPlugNodes(node, parent)

		# set default resultParams on node
		node.setParams(cls.defaultParams(Tree("root")))

		# make user-defined plugs
		inTree = Tree(cls.IN_PLUG_KEY)
		outTree = Tree(cls.OUT_PLUG_KEY)
		cls.makePlugs(inTree, outTree)
		#make plug nodes
		cls._makePlugNodesFromTree(inRoot, inTree, node)
		cls._makePlugNodesFromTree(outRoot, outTree, node)



	@classmethod
	def defaultParams(cls, paramRoot:Tree)->Tree:
		"""set up default resultParams for placing joints
		maybe pass in tree root as argument"""
		return paramRoot
