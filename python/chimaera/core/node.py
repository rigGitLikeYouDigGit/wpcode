
from __future__ import annotations
import typing as T

from wplib import Expression, DirtyExp, coderef
from wplib.sentinel import Sentinel
from wplib.object import UidElement, DirtyNode

from wptree import Tree



class ChimaeraNode(UidElement, DirtyNode):
	"""smallest unit of computation in chimaera graph

	refmap is key : node filter, not too crazy

	Simple node objects can also be compound graphs, with just a small
	modification to datablock.

	Graphs are nodes, some nodes are graphs.

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

	"""

	def __init__(self):
		UidElement.__init__(self)
		DirtyNode.__init__(self, self.clsName())
		self._parent : ChimaeraNode = None

		"""consider - this .data reference is never REPLACED - 
		this data dict may be embedded live into another graph,
		but this object won't know it.
		
		data can be looked up by UID, or from this attribute - 
		both point to the same dict object"""
		self.data = self.defaultData()

	# def _getDirtyNodeName(self) ->str:
	# 	return self.data["attrMap"]["name"].value

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
		refMapExp : T.Callable[[], dict[str, (str, ChimaeraNode)]] = DirtyExp(
			value={},
			name="refMap")
		attrMap["name"].setStructure("node")
		# nodes attribute cannot be proxied live -
		# if instancing needed, use a generator or directly reference
		# a parent graph node
		return {
			"attrMap" : attrMap,
			"refMap" : refMapExp,
			"nodes" : {},
		}

	@classmethod
	def setupNode(cls, node:ChimaeraNode, parent:ChimaeraNode)->None:
		"""default process to set up node when freshly created -
		used by plug nodes to create plugs, etc
		"""

	@classmethod
	def clsName(cls)->str:
		return cls.__name__

	def __str__(self):
		try:
			return f"<{self.clsName()}({self.nameExp()})"
		except:
			return f"<{self.clsName()}(UNABLE TO GET NAME - {self.getElementId()})>"


	def parent(self)->ChimaeraNode:
		return self._parent

	def dataBlock(self)->dict:
		"""should match exact object also tracked by parent graph"""
		#return self.parent().nodeDataBlock(self)
		return self.data

	def nameExp(self)->DirtyExp:
		return self.dataBlock()["attrMap"]["name"]
	def setName(self, name:str)->None:
		self.nameExp().setStructure(name)

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

	def setRef(self, key, uid:str="", nodeFilter:str=""):
		"""updates expression source of refmap with given value"""
		refMap = self.refMapExp().rawStructure()
		if uid:
			valueStr = f"uid:{uid}"
		refMap[key] = valueStr
		self.refMapExp().setStructure(refMap)

	def refMap(self)->dict[str, (str, ChimaeraNode)]:
		"""returns resolved refmap dict of nodes"""
		return self.refMapExp().eval()

	def refMapRaw(self)->dict[str, str]:
		"""returns raw refmap dict of strings"""
		return self.refMapExp().rawStructure()

	def getRef(self, key, default=Sentinel.FailToFind, raw=False):
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
		for childNode in nodes.values():
			nodes.update(childNode.allChildNodes())
		return nodes

	def addNode(self, node:ChimaeraNode,# nodeData:dict
	            ):
		"""add a node to the graph, with data"""
		#self.data["nodes"][node.getElementId()] = nodeData
		self.data["nodes"][node.getElementId()] = node.data
		node._parent = self

	@classmethod
	def defaultNodeType(cls)->T.Type[ChimaeraNode]:
		return cls

	def createNode(self, name:str, nodeType=None)->ChimaeraNode:
		"""create a node of type nodeType, add it to the graph, and return it"""
		nodeType = nodeType or self.defaultNodeType()
		newNode = nodeType()
		#newData = newNode.defaultData()
		newData = newNode.data
		self.addNode(newNode,# newData
		             )
		newNode.setName(name)
		return newNode
	#endregion

	# region node access
	def nodeByUid(self, uid:str)->ChimaeraNode:
		return ChimaeraNode.getByIndex(uid)

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
	def resolveRef(self, ref:str, fromNode:ChimaeraNode)->list[ChimaeraNode]:
		"""return sequence of nodes matching ref string -
		consider closer integration between this and node-side expressions.

		For test we only support single uid strings
		"""
		if ref.startswith("uid:"):
			uid = ref[4:]
			return [self.nodeByUid(uid)]
		if ref.startswith("n:"):
			name = ref[2:]
			return self.nodesByName(name)

	def resolveRefMap(self, refMap:dict, fromNode:ChimaeraNode)->dict:
		"""return dict of resolved references"""
		resolved = {}
		for key, ref in refMap.items():
			resolved[key] = self.resolveRef(ref, fromNode)
		return resolved


	# endregion


