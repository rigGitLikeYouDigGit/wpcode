
from __future__ import annotations
import typing as T

import fnmatch

from wplib import log
from wplib.uid import getUid4
from wplib.inheritance import clsSuper
from wplib.object import UidElement, ClassMagicMethodMixin

from wptree import Tree, lib as treelib



"""
can a node just be a tree?
outer node attributes each have "incoming" and "defined" ?
{
node uid : node name
	"type" : node type
	"params" : 
		"incoming" : "CLS"
		"defined" : "root" # could have root as value, to separate stable and dynamic regions
			
			 
	
	"storage" : 
		"root" 
	
	
currently graph structure depth is represented literally in tree data -
for now is ok


need to weaken connection between node wrapper and data - 
wrapper cannot be cached, since we need to dynamically
change the node type by string attribute

node.attr("type") -> attribute wrapper
node(".type") -> attribute wrapper
node.type -> attribute wrapper
node.type.resolve() -> resolved type
node.type.resolve()() -> new node of matching type

node.type() -> directly resolve attribute, resolved type
node.type()() -> directly resolve attribute, resolved type, new node of matching type


node("/type") -> child node called "type"


SEPARATE EVERYTHING.
separate node type looked up constantly is fine,
all concrete node objects just have python type ChimaeraNode 

Any filter or search operations return a flat list matching the query -
single values are taken as the first element, or None if no match or list is empty


node "type" attribute resolves to list - in the limit, a way to have nodes act as 
dynamic subclasses of these bases.
usually just a single value


child nodes tracked with "nodes" attr





incoming attributes sorted into tree with list values

list values composed together, and then child branches
are composed on top

root : [ a, b, c]
	branch : [ d.p, f.storage("branch2"), c ]



"""



def getEmptyTree():
	return Tree("root")


def getEmptyNodeAttributeData(name: str, incoming=("T", ), defined=())->Tree:
	t = Tree(name)
	t["incoming"] = list(incoming)
	t["defined"] = list(defined)
	return t


def composeIncomingTree(plugTree:Tree,
                        attrWrapper:NodeAttrWrapper,
                        parentNode:ChimaeraNode,
                        graph:ChimaeraNode)->Tree:
	"""plugTree has list branch values , which should each be lists of trees
	"""

	baseList = plugTree.value
	resultList = []
	nodeType : NodeType = parentNode.type()
	for i in baseList:
		# check for incoming value from node type
		if i == "T":
			resultList.append( nodeType.getTypeNodeAttrInput(attrWrapper.name()) )

	resultTree = treelib.overlayTrees(plugTree.value)
	# no recursion yet
	return resultTree




class NodeTypeBase(ClassMagicMethodMixin):
	"""Archetype for a kind of node -
	anything specific to a node's type should be defined here
	as class methods.

	Types shouldn't be instantiated, only inherited from.
	"""

	# region node type identification and retrieval

	nodeTypeRegister : dict[str, type[NodeTypeBase]] = {}

	@classmethod
	def prefix(cls)->tuple[str]:
		"""return the prefix for this node type -
		maybe use to define domain-specific node types.
		c : chimaera (always available)
		m : maya
		h : houdini
		b : blender
		n : nuke
		"""
		return ("c", )

	@classmethod
	def typeName(cls)->str:
		raise NotImplementedError

	@staticmethod
	def registerNodeType(cls:type[NodeTypeBase]):
		"""register the given node type -
		deal with prefixes some other time"""
		cls.nodeTypeRegister[cls.typeName()] = cls

	@classmethod
	def __init_subclass__(cls, **kwargs):
		super().__init_subclass__(**kwargs)
		cls.registerNodeType(cls)

	@staticmethod
	def getNodeType(lookup:str)->type[NodeTypeBase]:
		"""return the node type for the given lookup string.
		Later maybe allow searching somehow
		"""
		return NodeTypeBase.nodeTypeRegister[lookup]
	# endregion

	# region node attributes
	@classmethod
	def _getNewNodeAttrData(cls, attrName:str,
	                        incoming=("T", ),
	                        defined=())->Tree:
		"""return the data tree for a newly created node attribute -
		by default this is also the live default value for a node's evaluation"""
		empty = getEmptyNodeAttributeData(attrName, incoming=incoming, defined=defined)
		return empty

	@classmethod
	def getTypeNodeAttrInput(cls, attrName:str)->Tree:
		"""return the default input for a node attribute of this type"""
		return cls._getNewNodeAttrData(attrName)

	@classmethod
	def newNodeData(cls, name:str, uid="")->Tree:
		"""return the default data for a node of this type"""
		t = Tree(name)
		if uid:
			t.setElementId(uid)
		t.addChild(cls._getNewNodeAttrData("type", defined=(cls.typeName(),)
		                                   ))
		t.addChild(cls._getNewNodeAttrData("nodes", incoming="T"))
		#t.addChild( cls._getNewNodeAttrData("edges", incoming="T") )
		t.addChild(cls._getNewNodeAttrData("value"))
		t.addChild(cls._getNewNodeAttrData("params"))
		t.addChild(cls._getNewNodeAttrData("storage"))
		return t
	# endregion

	# region node creation

	@classmethod
	def create(cls, name:str, uid="")->ChimaeraNode:
		"""create a new node of this type"""
		print("create", name, uid)
		return ChimaeraNode(
			cls.newNodeData(name, uid=uid)
		)

	# @classmethod
	# def __class_call__(cls, name:str, *args, **kwargs)->ChimaeraNode:
	# 	"""create a new node of this type,
	# 	don't instantiate the type object itself"""
	# 	log("nodeType call", cls, name, args, kwargs)
	# 	return cls.create(name, *args, **kwargs)

	@staticmethod
	def __new__(cls, name:str, *args, **kwargs)->ChimaeraNode:
		try:
			cls.typeName()
		except NotImplementedError:
			raise NotImplementedError("Node type must define a typeName")
		log("nodeType call", cls, name, args, kwargs)
		return cls.create(name, *args, **kwargs)
	# endregion

class NodeType(NodeTypeBase):

	@classmethod
	def typeName(cls) ->str:
		return "base"



class NodeAttrWrapper:
	def __init__(self, attrData:Tree, node:ChimaeraNode):
		self._tree = attrData
		self.node = node

	def name(self)->str:
		return self._tree.name

	# region locally defined overrides
	def defined(self)->Tree:
		return self._tree("defined")

	def setDefined(self, value):
		"""manually override"""
		self._tree["defined"] = value
	#endregion

	# incoming connections
	def incomingTree(self)->Tree:
		"""tree with node connection values"""
		return self._tree("incoming")

	def incomingComposed(self)->Tree:
		"""composed final tree of incoming data"""
		return composeIncomingTree(self.incomingTree(),
		                           attrWrapper=self,
		                           parentNode=self.node,
		                           graph=self.node.parent())


	def setIncoming(self, value:(str, list[str])):
		"""reset incoming connections to given value"""

	# endregion

	# region resolved value
	def resolve(self)->Tree:
		"""return the resolved tree for this attribute.

		if defined is callable, call it with incoming data
		 and this node.
		if defined is tree, compose it with incoming and
		eval any expressions

		nodeType defines default behaviour in composition, which may
		be overridden at any level of tree
		"""
		incoming = self.incomingComposed()
		defined = self.defined()
		return treelib.overlayTrees([incoming, defined])



	def resolveToList(self)->list:
		"""return the resolved value of the top branch for this attribute
		"""
		val = self.resolve().value
		if isinstance(val, list):
			return val
		return [val]

	def __call__(self) -> Tree:
		# specialcase type, messy but whatever
		if self.name() == "type" :

			# T E M P
			# use directly defined type for now to avoid recursion
			return self.defined().value[0]

			return [NodeTypeBase.getNodeType(i) for i in self.defined().value + self.incomingTree().value][0]
		return self.resolve()

	# endregion

class ChimaeraNode(UidElement, ClassMagicMethodMixin):
	"""node's internal data is a tree -
	this wrapper may be created and destroyed at will.

	Node objects should look up their NodeType live from "type" attribute,
	so type can be changed dynamically.


	Tree composition behaviour -
		default defined by node type
		can be overridden within branches by expression
	"""

	# region uid registering
	indexInstanceMap = {} # global map of all initialised nodes


	def __init__(self, data:Tree=None):
		"""create a node from the given data -
		must be a tree with uid as name"""
		super().__init__(uid=data.uid)
		self._data : Tree = data

		# map just used for caching attribute wrappers, no hard data stored here
		self._attrMap : dict[str, NodeAttrWrapper] = {}

		# add attributes
		self.type = self._newAttrInterface("type")
		self.nodes = self._newAttrInterface("nodes")
		self.params = self._newAttrInterface("params")
		self.storage = self._newAttrInterface("storage")
		self.value = self._newAttrInterface("value")


	def _newAttrInterface(self, name:str)->NodeAttrWrapper:
		"""create a new interface wrapper for the given attribute name"""
		self._attrMap[name] = NodeAttrWrapper(self._data(name, create=False),
		                                      node=self)
		return self._attrMap[name]


	def getElementId(self) ->keyT:
		return self._data.uid

	@property
	def name(self)->str:
		return self._data.name

	def nodeTypeMRO(self)->list[type[NodeTypeBase]]:
		"""return the node type mro for this node"""
		return self.type.resolveToList()

	# region child nodes
	def parent(self)->ChimaeraNode:
		"""return the parent of this node"""
		if not self._data.parent: # top node of graph
			return None
		# parent of this graph will be the "nodes" branch -
		# we want the parent of that
		return self.indexInstanceMap[ self._data.parent.parent.uid ]

	def addNode(self, nodeData:(ChimaeraNode, Tree)):
		if isinstance(nodeData, ChimaeraNode):
			nodeData = nodeData._data

		assert nodeData.name not in self.nodes.defined().keys(), f"Node {nodeData.name} already exists in graph {self}"
		self.nodes.defined().addChild(nodeData)

	def children(self)->list[ChimaeraNode]:
		"""return the children of this node"""
		return [ChimaeraNode(i) for i in self.nodes.resolve().branches]

	def nameUidMap(self)->dict[str, str]:
		"""return a map of node names to uids"""
		return {n.name : n.uid for n in self._data("nodes").branches}
	#endregion

	# region edges
	# endregion


	@classmethod
	def __class_call__(cls, data:Tree)->ChimaeraNode:
		"""retrieve existing node for data, or instantiate new wrapper
		"""
		# check if node already exists
		lookup = cls.indexInstanceMap.get(data.uid, None)
		if lookup is not None:
			return lookup
		return type(clsSuper(cls)).__call__(cls, data)







if __name__ == '__main__':

	graph : ChimaeraNode = NodeType("graph")
	print(graph)

	newNode = ChimaeraNode(graph._data)
	print(newNode)
	print(newNode is graph)

	print(newNode.type())


