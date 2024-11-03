
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



"""
simplest version works with node name and UI fully reactive :D


QA:

Why don't we just have each tree be a node directly, and store all attributes as its value?
Why use the double layer ->tree->"nodes"->tree->"nodes" ?
because then to get a node's value, you would do node.value.value, or similar
node.value.resolved? node.value.incoming?
I guess it's not terrible 

node.data gives the core tree node - graph code shouldn't directly touch this
node.value gives all the attrs? 
node.value should be pure and simple result, the final data tree of the graph

node.result() - single method apart from all the other attribute processors
everything resolved and eval'd

OR:
we use attribute names to tell - 
@S - settings
@P - parent
@M - memory
@I - incoming



node initialisation again - 
for now, only do dynamics on base chimaeraNode class
ChimaeraNode(data) -> the best fitting node type for that data

"""



class ChimaeraNode(Modelled,
                   Pathable,
                   Visitable,
                   ClassMagicMethodMixin,
                   UidElement
                   ):
	keyT = Pathable.keyT
	pathT = Pathable.pathT

	@classmethod
	def dataT(cls):
		"""return type for modelled data
		TODO: later, allow this to be a combination of types,
			or a structural condition"""
		return Tree

	data : Tree | WpDexProxy

	@classmethod
	def newDataModel(cls, name="node",
	                 value=None,
	                 aux=None
	                 ) ->dataT():
		t = Tree(name=name, value=value,
		         )
		t["@S", "T"] = cls.typeName()
		#t.addBranch( Tree("nodes") ) #TODO: replace with the crazy incoming/defined etc
		return t

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


	# region uid registering
	indexInstanceMap = {} # global map of all initialised nodes
	@classmethod
	def getNodeType(cls, s:(ChimaeraNode, Tree, str))->type[ChimaeraNode]:
		a = 2
		if isinstance(s, ChimaeraNode):
			return cls.getNodeType(s.rawData()["@S", "T"])
		if isinstance(s, str):
			return cls.nodeTypeRegister.get(s)
		assert isinstance(s, Tree)
		# check if tree is settings
		if s.name == "@S":
			typeStr = s["T"]
			log("typeStr", typeStr, s)
			return cls.nodeTypeRegister.get(s)
		typeStr = s["@S", "T"]
		log("typeStr", typeStr, s)
		return cls.nodeTypeRegister.get(typeStr)


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
		"""init isn't ever called directly"""
		log("Chimaera init", data)
		assert isinstance(data, Tree)
		Modelled.__init__(self, data)
		Pathable.__init__(self, self, parent=None, name=data.name)
		UidElement.__init__(self, uid=data.uid)

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
	def parent(self) -> (Pathable, None):
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
	assert node.getNodeType(node.data) == CustomType
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















