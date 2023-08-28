
from __future__ import annotations
import typing as T

import pprint
#from collections import namedtuple
from typing import NamedTuple, TypedDict
from types import FunctionType


from wplib import DirtyExp, CodeRef
from wplib.expression import Expression, SyntaxPasses, ExpTokens, ExpSyntaxError, EvaluationError, ExpEvaluator, ExpPolicy, ExpSyntaxProcessor
from wplib.sentinel import Sentinel
from wplib.object import UidElement, DirtyNode
from wplib.sequence import getFirst, flatten

from wplib.serial import Serialisable, EncoderBase

from wptree import Tree

from chimaera.core.construct import NodeFnSet

"""how to pass the ref object parametres? How to store those parametres?

refMap{
	"refName" : "uid:adafas, affectEval:False"
}

has to resolve dynamically to dict - this might get quite messy :))) 

outer level is expression, evals 

"""

class RefValue(TypedDict):
	"""wrapper for ref object parametres"""
	uid : list[str]# = ()
	path : list[str]# = ()
	affectEval : int# = 1

def newRefValue() ->RefValue:
	return {"uid" : [], "path" : [], "affectEval" : 1}

class NodeExpEvaluator(ExpEvaluator):
	"""evaluate node expression"""

	def __init__(self, node:ChimaeraNode):
		self.node = node

	def resolveName(self, name:str):
		#print("nodeExpEvaluator resolveName", name)
		if name == "name":
			return self.node.name()
		if name == "uid":
			return self.node.uid
		if name == "value":
			return self.node.value()
		if name == "params":
			return self.node.resultParams()
		if name == "storage":
			return self.node.resultStorage()

#
# class EncoderBase(Serialisable.EncoderBase):
# 	"""base class for encoding Chimaera nodes"""


class ChimaeraNode(UidElement, DirtyNode, Serialisable):
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
	make ref map values proper function resultParams

	"""

	# special key in ref map pointing to construct class
	CONSTRUCT_CLS_REF_KEY = "_clsRef"

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
		self._evaluator = NodeExpEvaluator(self)
		self._expPolicy = self.getExpPolicy()
		self._data = self.makeDefaultData()

		# for now we give a node a fleeting reference to its active function set -
		# control flow is a bit tangled, but it's good enough for a v1
		self._fnSet : NodeFnSet = None




	#region dirtynode integration
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

	def dirtyComputeFn(self):
		if self.fnSetType():
			return self.fnSetType().dirtyComputeOuter(self, self.parent())
		return self.valueExp().eval()

	#endregion

	@classmethod
	def typeRefStr(cls)->str:
		"""return code ref to the type of this node,
		NOT to its construct type"""
		return CodeRef.get(cls)


	def fnSetTypeRefStr(self)->(str, None):
		"""return code ref to the construct type of this node
		weird to put this in refmap but it's the least likely
		thing to go missing from a node (and if you want to,
		you can always override it"""
		refValue = self.getRef(self.CONSTRUCT_CLS_REF_KEY, default=None)
		if refValue:
			return refValue["uid"][0]
		return refValue

	def setFnSetTypeRefStr(self, refStr:str):
		"""set code ref to the construct type of this node"""
		self.setRef(self.CONSTRUCT_CLS_REF_KEY, uid=(refStr,))

	def fnSetType(self)->T.Type[NodeFnSet]:
		"""return construct class of this node"""
		refStr = self.fnSetTypeRefStr()
		if refStr:
			return CodeRef.resolve(refStr)
		return None

	def setFnSet(self, fnSet:NodeFnSet=None):
		self._fnSet = fnSet
		self.setFnSetTypeRefStr(CodeRef.get(type(fnSet)))

	def fnSet(self)->NodeFnSet:
		return self._fnSet



	def makeDefaultData(self)->dict:
		"""called when node object is constructed,
		BEFORE any reference to graph is known"""
		attrMap : dict[str, Expression] = {
			"name" : DirtyExp(name="name",
			                  policy=self._expPolicy,
			                  evaluator=self._evaluator

			                  ),
			"value": DirtyExp(name="value",
			                  policy=self._expPolicy,
			                  evaluator=self._evaluator

			                  ),

			# putting these here for now for ease - only complex nodes need them,
			"resultParams" : DirtyExp(name="resultParams",
			                    value=Tree("root"),
			                          policy=self._expPolicy,
			                          evaluator=self._evaluator

			                          ),
			"storage" : DirtyExp(name="storage",
			                  value=Tree("root"),
								policy = self._expPolicy,
								 evaluator=self._evaluator

			                     ),
		}
		refMapExp : T.Callable[[], dict[str, dict[str]]] = DirtyExp(
			value={},
			name="refMap",
			policy=self._expPolicy,
			evaluator=self._evaluator

		)
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

	# region expression stuff
	def getExpPolicy(self)->ExpPolicy:
		"""return a policy object for Chimaera expression
		syntax and evaluation.
		Can't delegate to constructs since this directly affects how
		the graph structures itself.
		"""

		# tokenPass = SyntaxPasses.TokenReplacerPass(
		# 	tokenTypes=(ExpTokens.Dollar,)
		# )
		#
		def definesExpFn(s):
			return ExpTokens.Dollar.head in s

		charPass = SyntaxPasses.CharReplacerPass(
			charMap={ "$" : ExpPolicy.EXP_GLOBALS_KEY + ".evaluator." }
		)
		ensureLambdaPass = SyntaxPasses.EnsureLambdaPass()

		syntaxProcessor = ExpSyntaxProcessor(
			syntaxStringPasses=[charPass, ensureLambdaPass],
			syntaxAstPasses=[],
			stringIsExpressionFn=definesExpFn
		)

		evaluator = NodeExpEvaluator(self) # specialise this

		# def getExpGlobals():
		# 	return tokenPass.getSyntaxLocalMap()

		policy = ExpPolicy(
			syntaxProcessor=syntaxProcessor,
			evaluator=evaluator,
		)
		return policy

	def resolveExpNodeToken(self, token:str):
		if token == "name":
			return self.name()
		if token == "value":
			return self.value()
		if token == "params":
			return self.resultParams()
		if token == "storage":
			return self.resultStorage()

	# region name
	def nameExp(self)->DirtyExp:
		return self.dataBlock()["attrMap"]["name"]
	def sourceName(self)->str:
		return self.nameExp().rawStructure()
	def setName(self, name:str)->None:
		self.nameExp().setStructure(name)
	def name(self)->str:
		return self.nameExp().eval(self)

	#region resultParams
	def paramsExp(self)->DirtyExp:
		return self.dataBlock()["attrMap"]["resultParams"]
	def sourceParams(self)->Tree:
		return self.paramsExp().rawStructure()
	def setParams(self, value:Tree)->None:
		"""directly set the current value to rescan tree in expression"""
		self.paramsExp().setStructure(value)
	def resultParams(self)->Tree:
		return self.paramsExp().eval(self)
	# endregion
	
	#region storage
	def storageExp(self)->DirtyExp:
		return self.dataBlock()["attrMap"]["storage"]
	def sourceStorage(self)->Tree:
		return self.storageExp().rawStructure()
	def setstorage(self, value)->None:
		"""directly set the current value to rescan tree in expression"""
		self.storageExp().setStructure(value)
	def resultStorage(self)->Tree:
		return self.storageExp().eval(self)
	# endregion

	#region value and evaluation
	def valueExp(self)->DirtyExp:
		return self.dataBlock()["attrMap"]["value"]
	def sourceValue(self):
		return self.valueExp().rawStructure()
	def setValue(self, value)->None:
		self.valueExp().setStructure(value)
	def value(self):
		"""if exp is set, evaluate it
		if data tree is defined, return it,
		if resultParams tree is defined, return it?

		Really like old version where a node either operated on
		data or became data - maybe we can carry that through
		"""
		if self.fnSet():
			return self.fnSet().compute()
		if not self.valueExp().isEmpty():
			return self.valueExp().eval(self)
		return self.resultParams()

	# endregion

	#region refmap / node connections
	def refMapExp(self)->DirtyExp:
		return self.dataBlock()["refMap"]

	def setRef(self, key, uid:tuple[str, ...]=(), path:tuple[str, ...]=(), node:tuple[ChimaeraNode, ...]=(), affectEval=True, refVal:RefValue=None)->None:
		"""updates expression source of refmap with given value"""
		refMapSrc : dict = self.refMapExp().rawStructure()
		if refVal:
			refMapSrc[key] = refVal
		else:
			refMapSrc[key] = RefValue(
				uid=list(uid), path=list(path), affectEval=affectEval)
		self.refMapExp().setStructure(refMapSrc)

	def refMap(self)->dict[str, RefValue]:
		"""returns evaluated ref map expression - not yet
		resolved to chimaera nodes"""
		return self.refMapExp().eval()

	def refMapRaw(self)->dict[str, dict[str, RefValue]]:
		"""returns raw refmap dict of strings"""
		return self.refMapExp().rawStructure()

	def getRef(self, key, default:object=Sentinel.FailToFind, raw=False)->RefValue:
		"""return refmap entry for given key, or default if not found
		"""
		if raw:
			return self.refMapExp().rawStructure().get(key, default)
		result = self.refMap().get(key, default)
		if result is Sentinel.FailToFind:
			raise KeyError(f"refmap key {key} not found in {self.refMapRaw()}")
		return result
		baseRefValue = newRefValue()
		baseRefValue.update(result)
		return baseRefValue


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

	def createNode(self, name:str, nodeType:type[NodeFnSet]=None)->ChimaeraNode:
		"""create a node of type nodeTypeName, add it to the graph, and return it"""

		toCreateCls = ChimaeraNode


		#nodeType = nodeType or self.defaultNodeType()
		newNode = toCreateCls()
		#newData = newNode.makeDefaultData()
		newData = newNode.dataBlock()
		self.addNode(newNode,# newData
		             )


		# if construct class given, run node setup
		newNode.setName(name)
		if nodeType:
			nodeType.setupNode(newNode, name=name, parent=self)
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

	def resolveChildRef(self, ref:RefValue, fromNode:ChimaeraNode)->tuple[ChimaeraNode]:
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

	def resolveExpRef(self, localName:str, mode="value"):
		"""function called from within node expressions as REF()
		implement other modes to return node, to return literal text
		of ref node's value, etc
		Weird to call up to parent, see if there's a cleaner way
		"""
		return self.parent().resolveChildRef(self.getRef(localName), self)

	def getExpParseGlobals(self)->dict[str, object]:
		"""return dict of globals to be used when parsing expressions"""
		return {
			"REF" : self.resolveExpRef,
		}

	def getExpEvalGlobals(self)->dict[str, object]:
		"""return dict of globals to be used when evaluating expressions"""
		return {
			"REF" : self.resolveExpRef,
		}

	# endregion

	# region serialisation
	uniqueAdapterName = "ChimaeraNode"

	@Serialisable.encoderVersion(1)
	class Encoder(EncoderBase):
		"""encoder for Chimaera nodes"""

		@classmethod
		def encode(cls, obj: ChimaeraNode) -> dict:
			data = {}
			for k, v in obj.dataBlock().items():
				#print("k", k, "v", v, type(v))
				if isinstance(v, Expression):
					data[k] = v.rawStructure()
				else:
					data[k] = v
			#print("serial data for node", obj)
			#pprint.pprint(data)
			data["uid"] = obj.getElementId()
			return data

		@classmethod
		def decode(cls, serialCls: T.Type[ChimaeraNode], serialData: dict) -> ChimaeraNode:
			node = serialCls()
			node._data = serialData
			node.setElementId(serialData["uid"])
			return node



	# endregion
