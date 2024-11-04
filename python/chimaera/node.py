
from __future__ import annotations

import pprint
import typing as T

import fnmatch

from collections import namedtuple

import wplib.sequence
from wplib import log, Sentinel, TypeNamespace, Pathable
from wplib.constant import MAP_TYPES, SEQ_TYPES, STR_TYPES, LITERAL_TYPES, IMMUTABLE_TYPES
from wplib.object.visitor import PARAMS_T, CHILD_LIST_T
from wplib.uid import getUid4
from wplib.inheritance import clsSuper

from wplib.object import VisitAdaptor, Visitable, ClassMagicMethodMixin, UidElement


from wptree import Tree

from wpdex import *


from chimaera.lib import tree as treelib
from chimaera.attr import newAttributeData, NodeAttrWrapper, NodeAttrRef



"""
simplest version works with node name and UI fully reactive :D


QA:

Why don't we just have each tree be a node directly, and store all attributes as its value?
Why use the double layer ->tree->"nodes"->tree->"nodes" ?
because then to get a node's value, you would do node.value.value, or similar
node.value.resolved? node.value.linking?
I guess it's not terrible 

node.data gives the core tree node - graph code shouldn't directly touch this
node.value gives all the attrs? 
node.value should be pure and simple result, the final data tree of the graph

node.result() - single method apart from all the other attribute processors
everything resolved and eval'd

OR:
we use attribute names to tell - 
@S - settings
@M - memory
@F - flow (main graph data)


don't use the word "data", no one will know what it's referring to

graph               : root
	subgraph        : root
		leaf node   : root
						|- @S
						|- @M
						|- @F

but with THIS system, any nodes that generate other nodes need to reach back
across the value divide to modify graph structure
maybe that's ok 

IN GENERAL, define OVERRIDES at point of USE,
not point of CREATION
a node in principle should not know how its data will be used, and 
should not modify its own output to suit subsequent nodes in graph
(this can be bent but it seems good to keep complexity manageable)

-----
each attribute stores 2 trees 
2 STAGES of filtering data
"INCOMING" - tree of lists of references to other nodes or expressions of nodes;
			by default the first input of the root points to T, 
			takes the node type's default input
final INCOMING tree defines all real node connections (maybe also with refmap / blackboard) 

"OVERRIDE" - tree of manually-defined overrides over the linking tree VALUES
	Q: why not call it "defined" ? 
	A: because linking entries are also defined manually, override tree
		only affects resolved values in data
final OVERRIDE tree defines linking VALUES passed to COMPUTE

maybe later a "POST" stage, but that gets complicated if we want to change both structure
and values of the output data
the expected solution here is to just use a subsequent node to rearrange data in that
node's linking and override steps


RESOLVE INCOMING CONNECTIONS:
tree starts out with simple string values - list expressions, uids, etc

root : [ "T" ]
	+ trunk : [ "uid", "n:ctl*nodes", "uid.S[branch, leaf]" ]
		+ branch : [ "n:overrideNode"[trunk, branch, leaf] ]

expand all node expressions
resolve to tree[list[tuples]] of (node uid, attribute, path)

root : [ ("T", "value", ()) ]
	+ trunk : [ ("uid", "value", ()), 
					(ctlA uid, "F", ()),
					(ctlB uid, "F", ()),
					(other uid, "S", ("branch", "leaf")),
					 ]
		+ branch : [ (overrideNode uid, "value", ("trunk", "branch", "leaf")) ]

resolve THAT tree top-down, with lists resolving from left to right

root : [ type default value tree ]
	+ trunk : composite root value
		+ (composite tree from ctlA, ctlB, other)
		+ branch : overrideNode["trunk", "branch", "leaf"]
			+ (composite tree from overrideNode["trunk", "branch", "leaf"])
			
overlay those trees into final linking,
then apply OVERRIDE

then apply COMPUTE to get output

node initialisation dynamically gets node type for data
ChimaeraNode(data) -> the best fitting node type for that data

there is no good way to separate graph structure from param data, since evaluating params 
may generate new nodes, and nodes are saved as overrides on parent data

"""

def getEmptyNodeAttributeData(name: str, linking=("T", ), defined=())->Tree:
	t = Tree(name)
	t["linking"] = list(linking)
	t["override"] = list(defined)
	return t

class ChimaeraNode(Modelled,
                   Pathable,
                   Visitable,
                   ClassMagicMethodMixin,
                   UidElement
                   ):
	keyT = Pathable.keyT
	pathT = Pathable.pathT

	nodeTypeRegister : dict[str, type[ChimaeraNode]] = {}

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
		return "base" if cls is ChimaeraNode else cls.__name__

	@classmethod
	def canBeCreated(cls):
		"""use if you need to define abstract types of nodes"""
		return True

	@staticmethod
	def registerNodeType(cls:type[ChimaeraNode]):
		"""register the given node type -
		deal with prefixes some other time"""
		#cls.nodeTypeRegister[":".join((*cls.prefix(), cls.typeName()))] = cls
		cls.nodeTypeRegister[cls.typeName()] = cls
	@classmethod
	def __init_subclass__(cls, **kwargs):
		"""register a derived ChimaeraNode type"""
		super().__init_subclass__(**kwargs)
		if cls.canBeCreated():
			if "Proxy(" in cls.__name__: # weird interaction with generating proxy classes
				return
			cls.registerNodeType(cls)

	@classmethod
	def dataT(cls):
		"""return type for modelled data
		TODO: later, allow this to be a combination of types,
			or a structural condition"""
		return Tree

	data : Tree | WpDexProxy

	#region tree attributes
	@classmethod
	def newDataModel(cls, name="node",
	                 value=None,
	                 aux=None
	                 ) ->dataT():
		"""return the base data spine for a chimaera node
		since the type might be dynamic, place that first in tree

		"@T" defers to the default tree for that data, defined at
		class level and passed in on node evaluation -
		this way we only save deviations from the default
		"@T" will always be eval'd by the logic in ChimaeraNode
		"""
		t = Tree(name=name, value=value,
		         branches=(
		#t["@S", "T"] = cls.typeName()
		#t.value = Tree(name="V", value=Non
			#newAttributeData(name="@T", linking=(), override=(cls.typeName(), )), # type
			Tree("@T", value=None, branches=( # no hierarchy needed to resolve single type
				Tree("linking", value=()),
				Tree("override", value=(cls.typeName(), ))
			)),
			newAttributeData(name="@S", linking=("T", )), # settings
			newAttributeData(name="@F", linking=("T", )), # flow
			newAttributeData(name="@M", linking=("T", )), # memory
			newAttributeData(name="@NODES", linking=("T", )), # child nodes
		         ))
		return t

	@classmethod
	def defaultIncomingDataForAttr(cls, attrName:str, forNode:ChimaeraNode)->Tree:
		"""return the 'base state' to use for overlaying a node's overrides
		during evaluation
		OVERRIDE here"""
		if attrName == "@S" : return cls.defaultSettings(forNode)
		if attrName == "@M" : return cls.defaultMemory(forNode)
		if attrName == "@F" : return cls.defaultFlow(forNode)
		if attrName == "@NODES" : return None # special case
		return Tree("root")
	@classmethod
	def defaultSettings(cls, forNode:ChimaeraNode)->Tree:
		"""OVERRIDE to give the default parametre tree that this node will
		draw from"""
		return Tree("root")
	@classmethod
	def defaultMemory(cls, forNode:ChimaeraNode)->Tree:
		"""OVERRIDE to give default format for any dense data this node
		needs to store on file
		TODO: do we give each node a separate storage file for this,
			so we can copy data and structure independently between
			assets?"""
		return Tree("root")
	@classmethod
	def defaultFlow(cls, forNode:ChimaeraNode)->Tree:
		"""OVERRIDE
		this gives the default 'output' data that this node takes in to compute
		compute"""
		return Tree("root")


	# move all the methods for computing stages of attributes here
	# all of these work in place, so copy data at the start of your operation
	"""we end up with a lot of class methods, but this seems better than coupling
	each step straight to a library module.
	Also possible we could break up the objects more into an "attribute resolver"
	and the main node? 
	overcomplicated for now
	
	absolutely no idea if these should be instance or class methods - 
	going instance for now. sorry in advance to future ed
	
	"""
	# @classmethod
	# def resolveNodes(cls, expList:list[str], graph:ChimaeraNode):
	# 	return wplib.sequence.flatten(cls)

		# def resolveNodes(exp: str,
		#                  graph: ChimaeraNode,
		#                  fromNode: ChimaeraNode = None) -> list[ChimaeraNode]:
		# 	"""resolve a list of node expressions to nodes
		# 	for now just uids
		#
		# 	this doesn't add the node Type object for a "T" expression
		# 	"""
		# 	results = []
		# 	for i in exp:
		# 		# if i == "T":
		# 		# 	results.append(fromNode)
		# 		# 	continue
		# 		results.extend(graph.getNodes(i))
		# 	return graph.getNodes(exp)

	def resolveNodeExp(self, expStr:str)->tuple[str, str, Pathable.pathT]:
		"""return
		(node.uid, attrName, path in attribute)
		from expression"""

	def _expandLinkingTree(self, linkingTree: Tree,
	                       # #attrWrapper: NodeAttrWrapper,
	                       # parentNode: ChimaeraNode,
	                       # graph: ChimaeraNode=None
	                       ) -> None:
		"""expand the linking tree for this attribute -

		filtering actual tree with path requires explicitly
		defining tree to use - ".p[branch, leaf]" etc
		if .p or .v is not defined, will slice nodes found by
		uid matching

		STILL need a general solution to evaluate normal python code
		within these that RESOLVES to a node expression, path etc

		path is always () for now, get back to it later

		use THIS tree to determine node dependencies

		left side always resolves to nodes, midpoint always to an attr name,
		right side always to a tree path within that attribute

		TODO: we list nodes here, only keep their uids, then list again in populateExpandedTree

		return tree[list[tuple[
			str node uid,
			str attribute,
			tuple[str] path
			]]]"""
		# assert graph
		for branch in linkingTree.allBranches(includeSelf=True):
			rawValue: list[NodeAttrRef] = branch.value
			if rawValue is None:
				branch.value = []
				continue
			if not isinstance(rawValue, list):
				rawValue = [rawValue]
			resultTuples: list[NodeAttrRef] = []
			# log("raw value", rawValue)
			for exp in rawValue:  # individual string expressions

				# expand node lists to individual uids
				# print("EXPAND", i, i == "T",  i[0] == "T" if isinstance(i, tuple) else False)
				if exp == "T":
					resultTuples.append(NodeAttrRef("T", linkingTree.name, ()))
					continue
				if isinstance(exp, tuple):
					if not exp: continue
					if (exp[0] == "T"):
						resultTuples.append(NodeAttrRef("T", linkingTree.name, exp[2]))
						continue

				if isinstance(exp, str):
					# get the node uid, don't worry about anything else
					resultTuples.append(NodeAttrRef(exp, "@F", ()))
				if not isinstance(exp, tuple):
					raise NotImplementedError("can't do expressions yet")
				resultTuples.append(NodeAttrRef(*exp))
				# separate nodes / node terms from path if given
				# refs = [NodeAttrRef(node.uid, node.attr, node.path)
				#         for node in self.parent.access(
				# 		self.parent, i, )]
				#resultTuples.extend(refs)

			branch.value = resultTuples

	def _populateExpandedLinkingTree(self, expandedTree:Tree):
		# def populateExpandedTree(expandedTree: Tree[list[NodeAttrRef]],
		#                          attrWrapper: NodeAttrWrapper,
		#                          parentNode: ChimaeraNode,
		#                          graph: ChimaeraNode) -> Tree:
		"""populate the expanded tree with rich trees -
		expand each node attr ref into a rich tree"""

		for branch in expandedTree.allBranches(includeSelf=True):
			newValue = []
			for ref in branch.value:  # type:NodeAttrRef
				# log("populateExpandedTree", i, i.uid)

				if ref.uid == "T":
					newValue.append(self.defaultIncomingDataForAttr(
						expandedTree.name, self))
					continue
				# look at this beautiful line
				#newValue.append(self.parent.getNodes(i.uid)[0]._attrMap[i.attr].resolve()[i.path])
				foundNode : ChimaeraNode = self.parent.access(
					self.parent, ref.uid, one=True, values=False,
					uid=True
				)
				if not foundNode: continue
				newValue.append(
					foundNode.resolveAttribute(ref.attr)(ref.path)
				)

			branch.value = newValue

	def _collatePopulatedTree(
			self,
			populatedTree: Tree[list[Tree]])->Tree:
		"""overlay the populated tree -
		overlay each tree in populated branch value, left to right
		then for any child branches in populated tree,
		overlay the result branch at that path with the overlaid result
		"""
		resultTree = Tree(populatedTree.name)
		for populatedBranch in populatedTree.allBranches(includeSelf=True,
		                                                 depthFirst=True,
		                                                 topDown=True):
			address = populatedBranch.address(
				includeSelf=True, includeRoot=False, uid=False)
			# log(address)
			resultBranch = resultTree(
				address,
				create=True)
			for i in populatedBranch.value:
				resultBranch = treelib.overlayTreeInPlace(resultBranch, i,
				                                          mode="union")
		return resultTree

	# @classmethod
	# def _resolveAttrTrees(cls, linkingTree:Tree, overrideTree:Tree=None)->Tree:
	#
	# 	self._expandLinkingTree(t)
	# 	self._populateExpandedLinkingTree(t)
	# 	self._collatePopulatedTree(t)
	# 	treelib.overlayTreeInPlace(t, self.type.override().copy())
	def resolveAttribute(self, attr:(Tree, NodeAttrWrapper, str))->Tree:
		"""return the resolved tree for this attribute.

		if defined is callable, call it with linking data
		 and this node.
		if defined is tree, compose it with linking and
		eval any expressions

		nodeType defines default behaviour in composition, which may
		be overridden at any level of tree

		TODO: cache result and maybe the intermediate stages here
		"""

		#log("RESOLVE")
		if isinstance(attr, NodeAttrWrapper):
			atName = attr.name()
			attr = attr.tree
		elif isinstance(attr, str):
			atName = attr
		elif isinstance(attr, Tree):
			atName = attr.name
		else:
			raise RuntimeError("invalid input to resolve", attr, type(attr))

		if atName == "@T":
			# each step expanded for easier debugging
			t = self.type.linking().copy()
			self._expandLinkingTree(t)
			self._populateExpandedLinkingTree(t)
			self._collatePopulatedTree(t)
			treelib.overlayTreeInPlace(t, self.type.override().copy())
			return t # TODO: add in the proper stuff for multi-typing?
				# just as soon as literally one thing makes use of it

		if self.name() == "value":
			# send to nodeType compute
			return self.node.compute(
				self.resolveIncomingTree(
					self.linkingTreeExpanded()
				)
			)
		if not self.node.parent(): # unparented nodes can't do complex behaviour
			return self.override()
		linking = self.resolveIncomingTree(
			self.linkingTreeExpanded()
		)

		defined = self.override()

		return treelib.overlayTrees([linking, defined])

	def _consumeFirstPathTokens(self, path: pathT, **kwargs
	                            ) -> tuple[list[Pathable], pathT]:
		"""allow looking up by uids"""

		if kwargs.get("uid"):
			token, *path = path
			found = ChimaeraNode.getByIndex(token)
			if found:
				return [found], path
			found = Tree.getByIndex(token)
			if found:
				return [ChimaeraNode(found)], path
			raise self.PathKeyError("No uid found for", token, path)
		return super()._consumeFirstPathTokens(path, **kwargs)

	def computeFlow(self, data: Tree) -> Tree:
		"""main compute function of data - by default just overlay
		anything passed in as settings over flow

		linking
		expanded
		populated
		composed
		overridden
		-> compute
		output

		"""
		return treelib.overlayTrees(
			[data,
			 self.resolveAttribute("@S")]
		)

	def computeSettings(self, data: Tree) -> Tree:
		return data

	def computeMemory(self, data: Tree) -> Tree:
		return data

	#endregion

	# region uid registering
	indexInstanceMap = {} # global map of all initialised nodes
	@classmethod
	def getNodeType(cls, s:(ChimaeraNode, Tree, str))->type[ChimaeraNode]:
		a = 2
		origS = s
		if isinstance(s, ChimaeraNode):
			s = s.resolveAttribute("@T").v
			if isinstance(s, (tuple, list)):
				s = s[0]
		if isinstance(s, str):
			return cls.nodeTypeRegister.get(s)
		assert isinstance(s, Tree)
		# check if tree is settings
		if s.name != "@T": # check branches to find a @T entry
			for b in s.allBranches(includeSelf=False, depthFirst=False):
				if b.name == "@T":
					m = b.branchMap()
					log(m, type(m), m.keys(), "override" in m.keys())
					s = b.branchMap()["override"]
					break
			v = wplib.sequence.firstOrNone(s.value)
			return cls.nodeTypeRegister.get(v)

		if s.name == "@T":
			return cls.nodeTypeRegister.get(s.value)
		raise TypeError("Invalid input to get node type", s, origS)

	@classmethod
	def _nodeFromClsCall(cls,
	                     dataOrNodeOrName: (str, Tree, ChimaeraNode),
	                     uid: str = None,
	                     ) -> (ChimaeraNode, bool):
		"""create a node from the given data -
		handle all specific logic for retrieving from different params

		bool is whether node is new or not
		"""

		if isinstance(dataOrNodeOrName, ChimaeraNode):
			# get node type
			nodeType = cls.getNodeType(dataOrNodeOrName.rawData())
			if nodeType == cls:
				return dataOrNodeOrName, False

			return type(clsSuper(nodeType)).__call__(
				nodeType, dataOrNodeOrName.rawData()), False
			# return type(clsSuper(nodeType)).__call__(nodeType, dataOrNodeOrName[0].tree)


		if isinstance(dataOrNodeOrName, Tree):
			# check if node already exists
			lookup = cls.indexInstanceMap.get(dataOrNodeOrName.uid, None)
			if lookup is not None:
				return lookup, False

			# get node type
			nodeType = cls.getNodeType(dataOrNodeOrName)
			log("nodeType", nodeType)
			assert nodeType

			#return type(clsSuper(nodeType)).__call__(nodeType, dataOrNodeOrName[0])
			return type.__call__(nodeType, dataOrNodeOrName) , False

		if isinstance(dataOrNodeOrName, str):
			name = dataOrNodeOrName
			return cls.create(name)
			# data = cls.newNodeData(name, uid)
			# return type(clsSuper(cls)).__call__(cls, data), True

		if uid is not None:
			# check if node already exists
			lookupNode = cls.indexInstanceMap.get(uid, None)
			if lookupNode is not None:
				return lookupNode, False

			# create new node object around data
			lookupData = Tree.getByIndex(uid)
			if lookupData:
				return type(clsSuper(cls)).__call__(cls, lookupData), False

		raise ValueError(f"Must specify one of name, data or uid - invalid args \n {dataOrNodeOrName}, {uid} ")

	@classmethod
	def __class_call__(cls,
	                   dataOrNodeOrName:(str, Tree, ChimaeraNode),
	                   #uid:str=None,
	                   # parent=None
	                   )->ChimaeraNode:
		"""retrieve existing node for data, or instantiate new wrapper
				specify only one of:
		data - create object around existing data tree
		uid - retrieve existing data tree for object
		name - create new data tree with given name

		TODO: if called on explicit node type, with data/node of incompatible tyoe as arg,
		 raise error

		 priority is to pass through data to normal init process -
		 if it's a tree, look up its node type and pass it to Modelled
		"""
		# don't do any fancy processing if nodetype explicitly given
		log("cls call", cls, dataOrNodeOrName)
		if cls is not ChimaeraNode:
			return type.__call__(cls, dataOrNodeOrName# uid, parent
			                              )

		# if not any((dataOrNodeOrName, uid)):
		# 	raise ValueError("Must specify one of name, data or uid")
		# assert not all((dataOrNodeOrName, uid)), "Must specify only one of name, data or uid"

		node, isNew = cls._nodeFromClsCall(dataOrNodeOrName#, uid=uid
		                                   )
		if not isNew:
			return node
		# if parent is cls._MasterGraph: # don't parent top graph
		# 	return node
		#
		# if parent is None:
		# 	parent = cls.defaultGraph()
		# 	parent.addNode(node, force=True)
		# 	return node
		# parent.addNode(node)
		return node

	def getAvailableNodeTypes(self)->dict[str, type[ChimaeraNode]]:
		"""return a list of node types that this node can support as
		children - by default allow all registered types
		TODO: update this as a combined class/instance method
		"""
		return dict(self.nodeTypeRegister)

	def __init__(self, data:Tree):
		"""init isn't ever called directly by user,
		filtered by __new__ down to simple data tree"""
		log("Chimaera init", data)
		assert isinstance(data, Tree)
		Modelled.__init__(self, data)
		UidElement.__init__(self, uid=data.uid)
		# attribute wrappers
		self.type = NodeAttrWrapper(self.data("@T"), node=self)
		self.T = self.type
		self.settings = NodeAttrWrapper(self.data("@S"), node=self)
		self.S = self.settings
		self.memory = NodeAttrWrapper(self.data("@M"), node=self)
		self.M = self.memory
		self.flow = NodeAttrWrapper(self.data("@F"), node=self)
		self.F = self.flow
		self._nodes = NodeAttrWrapper(self.data("@NODES"), node=self)

		Pathable.__init__(self, self, parent=None, name=data.name)





	def childObjects(self, params:PARAMS_T) ->CHILD_LIST_T:
		"""maybe we should just bundle this in modelled
		I think for the core visit stuff we shouldn't pass in the proxy,
		might get proper crazy if we do

		"""
		results = VisitAdaptor.adaptorForObject(self.rawData()).childObjects(
			self.rawData(), params)
		return results

	@classmethod
	def newObj(cls, baseObj: Visitable, childDatas:CHILD_LIST_T, params:PARAMS_T) ->T.Any:
		"""this should just be a new copy of the data given
		TODO: here we constrain dataT() to be
		"""
		retrievedVisitor = VisitAdaptor.adaptorForType(cls.dataT())
		newData = cls.newDataModel(name=childDatas[0][1],
		                           value=childDatas[1][1])
		a = 4
		return cls(newData)


	def colour(self)->tuple[float, float, float]:
		return (0.2, 0.3, 0.7)

	@property
	def name(self) -> keyT:
		return self.data.getName()
	def setName(self, name: keyT):
		"""there's some really weird stuff around inheriting from Pathable,
		since Pathable calls this during its init - seems benign for now
		but watch for it"""
		#log("chim set name", name)
		self.data.setName(name)

	@property
	def parent(self) -> ChimaeraNode:
		"""parent data will be

		parentName :
			nodes :
				childName : etc
					nodes : etc

		intuitively, should the value of an attribute be
		the fully resolved version of that attribute?

		"""
		if not self.data.parent: return None

		return ChimaeraNode(self.data.parent)

	def _setParent(self, parent: Pathable):
		"""private as you should use addBranch to control hierarchy
		from parent to child - addBranch will call this internally"""
		return
	def addBranch(self, branch:Pathable, name:keyT=None):
		"""return the same object passed in"""
		assert isinstance(branch, (ChimaeraNode, Tree))
		if isinstance(branch, ChimaeraNode):
			data = branch.rawData()
		else:
			data = branch
		#log("node add branch", data)
		# add tree branch to this node's data
		self.data.addBranch(data)
		if isinstance(branch, ChimaeraNode): return branch
		return ChimaeraNode(data)

	def branchMap(self)->[keyT, ChimaeraNode]:
		return {name : ChimaeraNode(branch)
		        for name, branch in self.data.branchMap().items()
		        if not name.startswith("@")}

	if T.TYPE_CHECKING:
		def branches(self)->list[ChimaeraNode]: pass

	def createNode(self, nodeType:type[ChimaeraNode]=None, name="")->ChimaeraNode:
		log("createNode", nodeType, name)
		if isinstance(nodeType, str):
			nodeType = self.nodeTypeRegister.get(nodeType)
		nodeType = nodeType or ChimaeraNode
		name = name or nodeType.typeName()
		newNode = nodeType.create(name=name)
		log("newNode", newNode)
		self.addBranch(newNode, newNode.name)
		return newNode

	def onCreated(self, parent=None):
		"""called when node is newly created, and added
		to the given parent (if created within a graph)"""

	@classmethod
	def nodeForTree(cls, data:Tree):
		pass


	@classmethod
	def create(cls, name="node", **kwargs):
		newData = cls.newDataModel(name=name, **kwargs)
		#log("new data")
		#newData.display()
		#raise
		return cls(newData)

# register this base node type first
ChimaeraNode.registerNodeType(ChimaeraNode)

if __name__ == '__main__':

	class CustomType(ChimaeraNode):
		pass

	log(ChimaeraNode.nodeTypeRegister)

	node = CustomType.create()
	assert node.getNodeType(node) == CustomType
	dataT = node.getNodeType(node.data)
	log("dataT", dataT, type(dataT))
	assert dataT == CustomType
	log(node)
	log(ChimaeraNode.getNodeType(node.data))
	log(ChimaeraNode(node))





	# graph = ChimaeraNode.create("graph")
	# log(graph.path)
	# def t(*a):
	# 	log("GRAPH CHANGED", a)
	# graph.ref().rx.watch(t, onlychanged=False)
	# node = graph.createNode(name="childNode")
	# node = graph.createNode(name="childNodeB")
	#log("child nodes", graph.branches)















