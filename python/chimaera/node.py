
from __future__ import annotations
import typing as T

import fnmatch

from wplib import log
from wplib.uid import getUid4
from wplib.inheritance import clsSuper
from wplib.object import UidElement, ClassMagicMethodMixin

from wptree import Tree


from chimaera.lib import tree as treelib



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


example value incoming might be ["T", "uid", "uid.p[branch, leaf]"] -
so node type value is base, and incoming nodes are overlayed on top 

but surely we don't know what the node's default value is before knowing its inputs

defined value should be T.compute()

no reason to allow the tree-overlay stuff for value too - 

value should always be run through a callable or expression - that callable can
also handle overlaying params or parts of it, on value data before or after compute, etc

eventually a value expression should look like
"T.compute(self, $in)" or something

params might have sections of "preCompute" and "postCompute" -


still need the refMap for adding nodes into node-local variables

refMap : []
	"refName" : [node filters]


refMap : [node uid] <- inherit refMap from another node
	"refName" : [node filters]
	
	
incoming cannot edit incoming data - any logic is only used to filter and compose incoming

incoming:
	defined: raw tree of strings as entered by user
	resolved: tree of node lists, expressions eval'd
	evaluated: tree of rich trees in values
	composed: final tree of incoming graph data
	
	
value works on same principle as others - 
default is T.compute, which is a callable at root level, 
so operates on the root of all incoming data
	
	
	
so the tree stack would be:
overlay( value.incoming, params.resolve )

how can we generalise normal python functions to work on trees?
string searchReplace for example - 
MAPTONAMES(b, b.replace("L_", "R_"))
MAPTOVALUES()
MAPTOBRANCHES(b, b.name = b.name.replace("L_", "R_") )
?

with exp shorthand for variables?
($NAME=b): b.replace("L_", "R_")
($BRANCH=b): b.name.replace("L_", "R_") 

prefer this to the expressions, but should probably flip the arg names.
this flows better as a lemma to me, "let BRANCH equal b" etc,
but it's literally the opposite way round to the rest of python

(b=$NAME) : b.replace 

could always do simply $NAME = $NAME.replace("L_", "R_")

maybe something like the match statements?



intermediate stage of incoming tree should resolve branches to lists of tuples:

root : [(node uid, attribute, path), (node uid, attribute, path), (node uid, attribute, path)]
	+ branch : [(node uid, attribute), (node uid, attribute), (node uid, attribute)]
	
etc

"""



def getEmptyTree():
	return Tree("root")


def getEmptyNodeAttributeData(name: str, incoming=("T", ), defined=())->Tree:
	t = Tree(name)
	t["incoming"] = list(incoming)
	t["defined"] = list(defined)
	return t


def resolveIncomingTree(
		rawTree:Tree,
		attrWrapper:NodeAttrWrapper,
		parentNode:ChimaeraNode,
		graph:ChimaeraNode)->Tree:
	"""resolve the incoming tree for this attribute -
	replace string references with trees to compose"""
	baseList = rawTree.value
	newTree = rawTree.copy()
	resultList = []
	nodeType : NodeType = parentNode.type()
	for i in baseList:
		if isinstance(i, Tree):
			resultList.append(i)
			continue

		# check for incoming value from node type
		if i == "T":
			resultList.append( nodeType.getTypeNodeAttrInput(attrWrapper.name()) )
			continue

		# value may be "uid" or "uid.attr"
		# if "uid.attr", resolve attr
		# if "uid", resolve node
		uid, *attr = i.split(".")

		if graph is None: # nodes have no parent, only uids allowed
			try:
				nodes = [ChimaeraNode.getByIndex(uid)]
			except Exception as e:
				log(f"error resolving node uid {uid}")
				log(f"node {parentNode} has no parent, only node uids can be used for connections")
				raise e
		else:
			nodes = graph.getNodes(uid)
		if not attr: # default to value
			attr = "value"
		# resolve attributes to be used as input
		resultList.extend( [n._attrMap[attr].resolve() for n in nodes] )
	rawTree.value = resultList
	return rawTree




class _NodeTypeBase(ClassMagicMethodMixin):
	"""Archetype for a kind of node -
	anything specific to a node's type should be defined here
	as class methods.

	Types shouldn't be instantiated, only inherited from.
	"""

	# region node type identification and retrieval

	nodeTypeRegister : dict[str, type[_NodeTypeBase]] = {}

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
	def registerNodeType(cls:type[_NodeTypeBase]):
		"""register the given node type -
		deal with prefixes some other time"""
		cls.nodeTypeRegister[cls.typeName()] = cls

	@classmethod
	def __init_subclass__(cls, **kwargs):
		super().__init_subclass__(**kwargs)
		cls.registerNodeType(cls)

	@staticmethod
	def getNodeType(lookup:str)->type[_NodeTypeBase]:
		"""return the node type for the given lookup string.
		Later maybe allow searching somehow
		"""
		return _NodeTypeBase.nodeTypeRegister[lookup]
	# endregion

	@classmethod
	def compute(cls, node:ChimaeraNode#, inputData:Tree
	            )->Tree:
		""" OVERRIDE THIS

		active function of node, operating on incoming data.
		look up some specific headings of params if wanted -
		preIncoming, postIncoming, etc
		each of these can act to override at different
		points.

		If none is found, overlay all of params on value.

		The output of compute is exactly what comes out
		as a node's resolved value - if any extra overriding
		has to happen, do it here
		"""
		#log("base compute")
		#log("input")
		inputData = node.value.incomingTreeResolved()
		#inputData.display()

		#log("composed")
		inputData = node.value.incomingComposed()
		#inputData.display()

		assert isinstance(inputData, Tree)

		#node.params.resolve().display()


		return treelib.overlayTrees([inputData, node.value.defined()])

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
		#return cls._getNewNodeAttrData(attrName)
		return getEmptyTree()

	@classmethod
	def newNodeData(cls, name:str, uid="")->Tree:
		"""return the default data for a node of this type"""
		t = Tree(name)
		if uid:
			t.setElementId(uid)
		t.addChild(cls._getNewNodeAttrData("type",
		                                   incoming=(),
		                                   defined=(cls.typeName(),)
		                                   ))
		t.addChild(cls._getNewNodeAttrData("nodes",
		                                   defined=(),
		                                   incoming=("T",)
		                                   ))
		#t.addChild( cls._getNewNodeAttrData("edges", incoming="T") )
		t.addChild(cls._getNewNodeAttrData("value",
		                                   incoming=("T",),
		                                   defined=( ),
		                                   ))
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


	@staticmethod
	def __new__(cls, name:str, *args, **kwargs)->ChimaeraNode:
		try:
			cls.typeName()
		except NotImplementedError:
			raise NotImplementedError("Node type must define a typeName")
		log("nodeType call", cls, name, args, kwargs)
		return cls.create(name, *args, **kwargs)
	# endregion

class NodeType(_NodeTypeBase):

	@classmethod
	def typeName(cls) ->str:
		return "base"

	if T.TYPE_CHECKING:
		def __new__(cls, name: str, *args, **kwargs) -> ChimaeraNode:
			pass


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
	def incomingTreeRaw(self)->Tree:
		"""raw tree with string filters"""
		return self._tree("incoming", create=False)

	def incomingTreeResolved(self)->Tree:
		"""incoming with rich trees resolved"""
		return resolveIncomingTree(self.incomingTreeRaw(),
		                           self,
		                           self.node,
		                           self.node.parent())

	def incomingComposed(self)->Tree:
		"""composed final tree of incoming data"""

		baseTree = self.incomingTreeResolved()
		resultList = baseTree.value
		# overlay all incoming trees for this level of input tree
		# TODO: proper node EvaluationReport / error system to give details
		assert all(isinstance(i, Tree) for i in
		           resultList), f"resultList {resultList} \n for attr {self} not all trees"
		try:
			resultTree = treelib.overlayTrees(resultList)
			# no recursion yet
			return resultTree
		except Exception as e:
			log("error composing incoming tree")
			self.incomingTreeResolved().display()
			raise e


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

		if self.name() == "value":
			# send to nodeType compute
			return self.node.type().compute(
				self.node,
				#self.incomingComposed()
			)

		incoming = self.incomingComposed()

		defined = self.defined()

		return treelib.overlayTrees([incoming, defined])

		try:
			return treelib.overlayTrees([incoming, defined])
		except Exception as e:
			log("error overlaying incoming and defined")
			log("incoming")
			incoming.display()
			log("defined")
			defined.display()
			raise e

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
			return _NodeTypeBase.getNodeType(self.defined().value[0])

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
		if isinstance(data, ChimaeraNode):
			data = data._data
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

	def nodeTypeMRO(self)->list[type[_NodeTypeBase]]:
		"""return the node type mro for this node"""
		return self.type.resolveToList()

	# region child nodes
	def parent(self)->ChimaeraNode:
		"""return the parent of this node

		graph stores nodes as attributes, so

		graphRoot
			+ nodes
				+ defined
					nodeA
					nodeB
				+ incoming
					nodeC

		parent.parent.parent is graphRoot

		"""
		if not self._data.parent: # top node of graph
			return None
		# parent of this graph will be the "nodes" branch -
		# we want the parent of that

		return ChimaeraNode(self._data.parent.parent.parent )

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

	def nodeMap(self)->dict[str, ChimaeraNode]:
		combinedMap = {}
		for data in self.nodes.resolve().branches:
			combinedMap[data.name] = ChimaeraNode(data)
			combinedMap[data.uid] = ChimaeraNode(data)
		return combinedMap

	def getNodes(self, pattern:str)->list[ChimaeraNode]:
		"""return all nodes matching the given pattern -
		combine names and uids into map, match against keys"""
		matches = fnmatch.filter(self.nodeMap().keys(), pattern)
		return [self.nodeMap()[i] for i in matches]


	# def getNode(self, node:(str, ChimaeraNode, Tree)):
	# 	if isinstance(node, str):
	# 		uidCheck = self.indexInstanceMap.get(node)
	# 		if uidCheck is not None:
	# 			return uidCheck
	# 		# check for name match
	# 		nameCheck = self.nameUidMap()
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

	#print(newNode.type())

	# log("data")
	# graph._data.display()
	#
	# log("value defined")
	# graph.value.defined().display()

	# #print(graph.value.resolve())
	# value = graph.value.resolve()
	# log("final value")
	# value.display()

	# log("incomingRaw")
	# graph.value.incomingTreeRaw().display()
	# log("incomingResolved")
	# graph.value.incomingTreeResolved().display()


	nodeA = NodeType("nodeA")
	nodeB = NodeType("nodeB")
	graph.addNode(nodeA)

	assert nodeA.parent() is graph
	graph.addNode(nodeB)


	# nodes = graph.getNodes("node*")
	# print(graph.getNodes("node*"))
	#
	# print(nodeA in nodes)
	# print(nodeB in nodes)

	# set up string join operation
	nodeA.value.defined().value = "start"
	nodeB.value.defined().value = "end"

	class StringJoinOp(NodeType):

		@classmethod
		def compute(cls, node:ChimaeraNode#, inputData:Tree
	            ) ->Tree:
			"""active function of node, operating on incoming data.
			join incoming strings
			"""
			log("string op compute")
			joinToken = node.params()["joinToken"]

			# this could be done with just a list connection to single
			# tree level, but this is more descriptive
			incoming = node.value.incomingTreeResolved()
			aValue = incoming["a"]
			bValue = incoming["b"]
			result = aValue + joinToken + bValue
			return Tree("root", value=result)


	opNode : ChimaeraNode = StringOp(name="opNode")
	graph.addNode(opNode)

	# connect nodes
	opNode.value.incomingTreeRaw()["a"] = nodeA.uid
	opNode.value.incomingTreeRaw()["b"] = nodeB.uid

	# get result
	result = opNode.value.resolve()

	log("result")














