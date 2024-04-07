
from __future__ import annotations
import typing as T
import json, sys, os
from pathlib import Path
from typing import TypedDict
from collections import defaultdict
from dataclasses import dataclass

import orjson

from wptree import Tree

from wpm import WN, om, cmds
from wpm.core import getMFn


TARGET_NODE_DATA_PATH = Path(__file__).parent / "nodeData.json"

@dataclass
class AttrData:
	"""data for a single attribute"""
	name:str
	type:int
	typeName:str
	isArray:bool
	#isCompound:bool
	# children:tuple[AttrData] = ()
	isInput:bool = False
	isOutput:bool = False
	shortName:str = ""
	parentName:str = "" # parent attribute name if attribute is a child
	# isInternal:bool = False

def getAttrData(obj:om.MObject):
	baseFn : om.MFnAttribute = getMFn(obj)
	#print("type", baseFn.name())

	data = AttrData(
		name=baseFn.name,
		type=baseFn.type(),
		#typeName=str(type(baseFn)),
		typeName=baseFn.object().apiTypeStr(),
		isArray=baseFn.array,
		shortName=baseFn.shortName,
		isInput=baseFn.writable,
		isOutput=baseFn.readable,
		#isCompound=baseFn
	)

	if baseFn.parent != om.MObject.kNullObj:
		data.parentName = om.MFnAttribute(baseFn.parent).name

	return data

@dataclass
class NodeData:
	"""data for a single node type"""
	typeName:str
	apiTypeConstant:int
	bases: tuple[str] = ()
	isAbstract:bool = False


	#attrIds:tuple[int]
	attrDatas:T.List[AttrData] = ()

	#attrNames:tuple[str] = ()

def getNodeData(nodeTypeName:str):

	mclass = om.MNodeClass(nodeTypeName)

	# get node class bases
	baseClasses = cmds.nodeType(nodeTypeName, isTypeName=1, inherited=1) or []
	baseClasses = ["_BASE_"] + baseClasses[:-1]  # remove the last one, which is the node type itself

	# get type ids for all attributes
	try:
		attrs : list[om.MObject] = mclass.getAttributes()
	except:
		print("failed to get attributes for", nodeTypeName)
		raise
	# get all attribute data
	attrDatas =list(sorted((getAttrData(attr) for attr in attrs), key=lambda x: x.name))

	data = NodeData(
		typeName=nodeTypeName,
		apiTypeConstant=int(mclass.typeId.id()),
		#attrs=mclass.attributeList()
		attrDatas=attrDatas,
		bases=tuple(baseClasses),
	)
	return data


def getBaseNodeData():
	"""return a node data representing the base of a node,
	just takes the common evaluation attributes"""
	cachingData = AttrData(
		name="caching",
		type=om.MFn.kNumericAttribute,
		typeName="om.MFnNumericAttribute",
		isArray=False,
		isInput=True, isOutput=True,
		shortName="cch"
	)
	frozenData = AttrData(
		name="frozen",
		type=om.MFn.kNumericAttribute,
		typeName="om.MFnNumericAttribute",
		isArray=False,
		isInput=True, isOutput=True,
		shortName="frz"
	)




	isHistoricallyInterestingData = AttrData(
		name="isHistoricallyInteresting",
		type=om.MFn.kNumericAttribute,
		typeName="om.MFnNumericAttribute",
		isArray=False,
		isInput=True, isOutput=True,
		shortName="hst",
	)

	messageData = AttrData(
		name="message",
		type=om.MFn.kMessageAttribute,
		typeName="om.MFnMessageAttribute",
		isArray=False,
		isInput=True, isOutput=True,
		shortName="msg"
	)

	nodeStateData = AttrData(
		name="nodeState",
		type=om.MFn.kEnumAttribute,
		typeName="om.MFnEnumAttribute",
		isArray=False,
		isInput=True, isOutput=True,
		shortName="nst",
	)


	return NodeData(
		typeName="_BASE_",
		apiTypeConstant=0,
		attrDatas=[ messageData, cachingData, frozenData,
		            isHistoricallyInterestingData, nodeStateData ],
		bases=(),
		isAbstract=True,
	)


def gatherNodeData():
	"""gather node data from maya
	"""
	# get all node types

	nodeTypes = cmds.allNodeTypes(includeAbstract=1)
	# this adds " (abstract)" to the end of abstract node types :)
	abstractMap = {x.replace(" (abstract)", ""): " (abstract)" in x for x in nodeTypes}
	nodeTypes = [x.replace(" (abstract)", "") for x in nodeTypes]


	# recover type tree for all nodes, avoid duplication

	typeLenSetMap : dict[int, set[str]] = defaultdict(set)

	for nodeType in nodeTypes:
		# get node class bases
		baseClasses = cmds.nodeType(nodeType, isTypeName=1, inherited=1) or []
		baseClasses = baseClasses[:-1]  # remove the last one, which is the node type itself
		# all nodes have "frozen", "message" etc attributes, create a "_BASE_" node type for these
		baseClasses.insert(0, "_BASE_")
		typeLenSetMap[len(baseClasses)].add(nodeType)

	nodeData : dict[str, dict[str, NodeData] ]= {}

	nodeData["0"] = {"_BASE_" : getBaseNodeData() }

	for typeLen, typeSet in sorted(typeLenSetMap.items(), key=lambda x: x[0]):
		nodeData[str(typeLen)] = {}
		#for nodeType in sorted(typeSet, key=lambda x: x):
		for nodeType in sorted(typeSet):
			data = getNodeData(nodeType)
			data.isAbstract = abstractMap[nodeType]
			nodeData[str(typeLen)][nodeType] = data



	# write to file
	with open(TARGET_NODE_DATA_PATH, "w") as f:
		f.write(
			orjson.dumps(nodeData, option=orjson.OPT_INDENT_2).decode("utf-8")
		)

"""
script:

from importlib import reload
from wpm.core.node.gen import gather

reload(gather)

gather.gatherNodeData()


"""


# wow hikHandle is a bolshy node

