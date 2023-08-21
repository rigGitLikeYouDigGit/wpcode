
from __future__ import annotations
import typing as T

#from collections import namedtuple
from typing import NamedTuple, TypedDict

from wplib import Expression, DirtyExp, coderef
from wplib.sentinel import Sentinel
from wplib.object import UidElement, DirtyNode

from wptree import Tree

from chimaera.core.construct import NodeConstruct

"""how to pass the ref object parametres? How to store those parametres?

refMap{
	"refName" : "uid:adafas, affectEval:False"
}

has to resolve dynamically to dict - this might get quite messy :))) 

outer level is expression, evals 

"""

class RefValue(TypedDict):
	"""wrapper for ref object parametres"""
	uid : tuple[str]# = ()
	path : tuple[str]# = ()
	affectEval : int# = 1

# class RefObject:
# 	"""wrapper for object that can be referenced in graph
# 	"""
# 	def __init__(self, obj:T.Any):
# 		self.obj = obj


class ChimaeraNode(UidElement, DirtyNode):
	"""smallest unit of computation in chimaera graph

	refmap is key : node filter, not too crazy

	Simple node objects can also be compound graphs, with just a small
	modification to datablock.

	Graphs are nodes, non-leaf nodes are graphs.

	In this model the parent/child relationship is rock solid,
	and not defined by node edges.

	Each subgraph has to manage its own stored data, which could be
	more difficult - however it means that any graph only has
	to be concerned with its direct children,
	it makes it way easier to merge and reference parts of the graph, etc.

	ON BALANCE, I think the tree hierarchy model is better for now. If needed maybe we can add some way to emulate subgraphs over a flat hierarchy.

	Compound nodes hold datablocks for all contained nodes?

	for test, we also inherit from DirtyNode - may separate to an
	execution-specific wrapper later

	not all features of the graph can (or should) appear in the execution graph -
	make ref map values proper function params

	"""

	def __init__(self, debugName:str=""):
		UidElement.__init__(self)
		DirtyNode.__init__(self, debugName)
		self._parent : ChimaeraNode = None
		self._debugName = debugName

		"""consider - this .data reference is never REPLACED - 
		this data dict may be embedded live into another graph,
		but this object won't know it.
		
		use datablock() getter EVERYWHERE, it does more calls to graph
		but it's safe to copied or instanced data -
		only falls back to this object's attribute if no parent is available
		"""
		self._data = self.defaultData()

	# def _getDirtyNodeName(self) ->str:
	# 	return self._data["attrMap"]["name"].value

	def getDirtyNodeAntecedents(self) ->tuple[DirtyNode]:
		"""return all nodes that this node depends on -
		return expressions directly connected to this
		node, parents will handle legitimate edges in graph
		"""
		return (
			self.nameExp(),
			self.valueExp(),
			self.refMapExp()
		)

	@classmethod
	def typeRefStr(cls)->str:
		return coderef.getCodeRef(cls)

	@classmethod
	def defaultData(cls)->dict:
		"""called when node object is constructed,
		BEFORE any reference to graph is known"""
		attrMap : dict[str, Expression] = {
			"name" : DirtyExp(name="name"),
			"value": DirtyExp(name="value"),
		}
		refMapExp : T.Callable[[], dict[str, dict[str]]] = DirtyExp(
			value={},
			name="refMap")
		attrMap["name"].setStructure("node")
		# nodes attribute cannot be proxied live -
		# if instancing needed, use a generator or directly reference
		# a parent graph node

		# a second "message" ref map is guaranteed not to affect evaluation -
		# these edges are not picked up by the execution graph,
		# and only appear in the data structure

		return {
			"attrMap" : attrMap,
			"refMap" : refMapExp,
			"nodes" : {},
		}

	@classmethod
	def clsName(cls)->str:
		return cls.__name__

	def __str__(self):
		try:
			return f"<{self.clsName()}-{self._debugName}({self.nameExp()})"
		except:
			return f"<{self.clsName()}-{self._debugName}(UNABLE TO GET NAME - {self.getElementId()})>"


	def parent(self)->ChimaeraNode:
		return self._parent

	def dataBlock(self)->dict:
		"""should match exact object also tracked by parent graph"""
		#return self.parent().nodeDataBlock(self)
		# if self.parent():
		# 	return self.parent().nodeDataBlock(self)
		return self._data

	def nameExp(self)->DirtyExp:
		return self.dataBlock()["attrMap"]["name"]
	def setName(self, name:str)->None:
		self.nameExp().setStructure(name)
	def nodeName(self)->str:
		return self.nameExp().resultStructure()

	#region value and evaluation
	def valueExp(self)->DirtyExp:
		return self.dataBlock()["attrMap"]["value"]
	def setValue(self, value)->None:
		self.valueExp().setStructure(value)

	def value(self):
		"""if exp is set, evaluate it
		if data tree is defined, return it,
		if params tree is defined, return it?

		Really like old version where a node either operated on
		data or became data - maybe we can carry that through
		"""
		if self.valueExp().rawStructure():
			return self.valueExp().resultStructure()

	# endregion

	#region refmap / node connections
	def refMapExp(self)->DirtyExp:
		return self.dataBlock()["refMap"]

	def setRef(self, key, uid:tuple[str]=(), path:tuple[str]=(), node:tuple[ChimaeraNode]=(), affectEval=True)->None:
		"""updates expression source of refmap with given value"""
		refMapSrc : dict = self.refMapExp().rawStructure()
		refMapSrc[key] = RefValue(uid=uid, path=path, affectEval=affectEval)
		self.refMapExp().setStructure(refMapSrc)

	def refMap(self)->dict[str, RefValue]:
		"""returns evaluated ref map expression - not yet
		resolved to chimaera nodes"""
		return self.refMapExp().eval()

	def refMapRaw(self)->dict[str, dict[str, RefValue]]:
		"""returns raw refmap dict of strings"""
		return self.refMapExp().rawStructure()

	def getRef(self, key, default=Sentinel.FailToFind, raw=False)->RefValue:
		"""return refmap entry for given key, or default if not found
		"""
		if raw:
			return self.refMapExp().rawStructure().get(key, default)
		return self.refMap().get(key, default)


	#endregion

	#region child nodes
	def isNetwork(self)->bool:
		"""return true if this node is a network"""
		return bool(self.dataBlock()["nodes"])

	def childNodes(self)->dict[str, ChimaeraNode]:
		"""return dict of child nodes"""
		return {k : ChimaeraNode.getByIndex(k)
		        for k, v in self.dataBlock()["nodes"].items()}

	def allChildNodes(self)->dict[str, ChimaeraNode]:
		"""return dict of child nodes, and their children, etc"""
		nodes = self.childNodes()
		for childNode in tuple(nodes.values()):
			nodes.update(childNode.allChildNodes())
		return nodes

	def addNode(self, node:ChimaeraNode,# nodeData:dict
	            ):
		"""add a node to the graph, with data"""
		self._data["nodes"][node.getElementId()] = node.dataBlock()
		node._parent = self

	@classmethod
	def defaultNodeType(cls)->T.Type[ChimaeraNode]:
		return ChimaeraNode

	def createNode(self, name:str, nodeType:type[NodeConstruct]=None)->ChimaeraNode:
		"""create a node of type nodeTypeName, add it to the graph, and return it"""

		toCreateCls = ChimaeraNode


		#nodeType = nodeType or self.defaultNodeType()
		newNode = toCreateCls()
		#newData = newNode.defaultData()
		newData = newNode.dataBlock()
		self.addNode(newNode,# newData
		             )


		# if construct class given, run node setup
		if nodeType:
			nodeType.setupNode(newNode, name=name, parent=self)
		else: # just set basic name
			newNode.setName(name)
		return newNode
	#endregion

	# region node access
	def nodeByUid(self, uid:str)->ChimaeraNode:
		return ChimaeraNode.getByIndex(uid)

	def iterDataBlocks(self)->T.Iterator[tuple[str, dict]]:
		"""iterate over all data blocks in this node and child nodes"""
		for uid, data in self.dataBlock()["nodes"].items():
			yield uid, data
			# for uid, data in ChimaeraNode.getByIndex(uid).iterDataBlocks():
			# 	yield uid, data

	def nodesByName(self, nameStr:str)->list[ChimaeraNode]:
		"""return list of nodes matching name"""
		names = nameStr.split(" ")
		nodes = []
		for uid, data in self.iterDataBlocks():
			if data["attrMap"]["name"].resultStructure() in names:
				nodes.append(self.nodeByUid(uid))
		return nodes

	# endregion

	# region node referencing and connections
	@classmethod
	def templateNodeRefDict(cls)->dict[str]:
		"""return a template for the dict defining a single reference"""
		return {
			# return nodes by uid
			"uid": "",

			# refer to nodes by path
			"path": "",

			# should this edge appear in execution graph?
			"affectEval" : True,
		}

	def resolveRef(self, ref:RefValue, fromNode:ChimaeraNode)->tuple[ChimaeraNode]:
		"""return sequence of nodes matching ref strings -
		consider closer integration between this and node-side expressions.

		RefValue is a dict with keys:
		uid: return nodes by uid
		path: refer to nodes by path
		affectEval: should this edge appear in execution graph?

		each of the node categories are tuples, may contain multiple string
		values - each string value must be evaluated and results combined to
		get all nodes matching the ref

		"""
		nodes = set()
		for i in ref["uid"]:
			nodes.add(self.nodeByUid(i))
		for i in ref["path"]:
			nodes.update(self.nodesByName(i))

		return tuple(nodes)

	# def resolveRefMap(self, refMap:dict, fromNode:ChimaeraNode)->dict:
	# 	"""return dict of resolved references"""
	# 	resolved = {}
	# 	for key, ref in refMap.items():
	# 		resolved[key] = self.resolveRef(ref, fromNode)
	# 	return resolved


	# endregion


