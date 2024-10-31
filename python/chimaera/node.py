
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
from wplib.object import UidElement, ClassMagicMethodMixin, CacheObj
from wplib.serial import Serialisable
#from wplib.pathable import Pathable
from wplib.object import VisitAdaptor, Visitable


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
"""



class ChimaeraNode(Modelled,
                   Pathable,
                   Visitable,
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
		return "node" if cls is ChimaeraNode else cls.__name__

	@classmethod
	def canBeCreated(cls):
		"""use if you need to define abstract types of nodes"""
		return True

	@staticmethod
	def registerNodeType(cls:type[ChimaeraNode]):
		"""register the given node type -
		deal with prefixes some other time"""
		cls.nodeTypeRegister[":".join((*cls.prefix(), cls.typeName()))] = cls
	@classmethod
	def __init_subclass__(cls, **kwargs):
		"""register a derived ChimaeraNode type"""
		super().__init_subclass__(**kwargs)
		if cls.canBeCreated():
			if "Proxy(" in cls.__name__: # weird interaction with generating proxy classes
				return
			cls.registerNodeType(cls)


	def getAvailableNodeTypes(self)->dict[str, type[ChimaeraNode]]:
		"""return a list of node types that this node can support as
		children - by default allow all registered types
		TODO: update this as a combined class/instance method
		"""
		return dict(self.nodeTypeRegister)

	def __init__(self, data:Tree):
		assert isinstance(data, Tree)
		Modelled.__init__(self, data)
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
		        for name, branch in self.data.branchMap().items()}

	def createNode(self, nodeType:type[ChimaeraNode]=None, name="")->ChimaeraNode:
		log("createNode", nodeType, name)
		if isinstance(nodeType, str):
			nodeType = self.nodeTypeRegister.get(nodeType)
		nodeType = nodeType or ChimaeraNode
		name = name or nodeType.typeName()
		newNode = nodeType.create(name=name)
		self.addBranch(newNode, newNode.name)
		return newNode

	@classmethod
	def nodeForTree(cls, data:Tree):
		pass


	@classmethod
	def create(cls, name="node", **kwargs):
		return cls(cls.newDataModel(name=name, **kwargs))


if __name__ == '__main__':


	graph = ChimaeraNode.create("graph")
	log(graph.path)
	def t(*a):
		log("GRAPH CHANGED", a)
	graph.ref().rx.watch(t, onlychanged=False)
	node = graph.createNode(name="childNode")
	node = graph.createNode(name="childNodeB")
	#log("child nodes", graph.branches)















