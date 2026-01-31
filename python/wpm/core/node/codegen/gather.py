
from __future__ import annotations

import pprint
import traceback
import typing as T
import json, sys, os
from pathlib import Path
from typing import TypedDict
from collections import defaultdict
from dataclasses import dataclass

import orjson

from wplib import log
from wptree import Tree

from wpm import WN, om, cmds
from wpm.core import getMFn, getMFnType, apiTypeMap, getCache

"""orjson does not output non-string keys, even when explicitly given
the flag. oh well."""

TARGET_NODE_DATA_PATH = Path(__file__).parent / "nodeData.json"
PLUGIN_NODE_DATA_PATH = Path(__file__).parent / "pluginNodeData.json"

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

	#todo: get the data type constants where possible - k2float, kTime, etc

def getAttrData(obj:om.MObject):
	baseFn : om.MFnAttribute = getMFn(obj)
	#print("type", baseFn.name())

	data = AttrData(
		name=baseFn.name,
		type=baseFn.type(),
		#typeName=str(type(baseFn)),
		typeName=baseFn.object().apiTypeStr,
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
	apiTypeStr : str # MFnTransform
	typeIdInt : int
	mfnStr : str
	bases: tuple[str] = ()
	isAbstract:bool = False


	#attrIds:tuple[int]
	attrDatas:T.List[AttrData] = ()

	#attrNames:tuple[str] = ()

reportTypes = {"nurbsCurve", "transform", "lambert", "animLayer",
               "bakeSet", "anisotropic"}

def getNodeData(nodeTypeName:str):

	#log("get data for ", nodeTypeName)
	def pr(*s):
		if nodeTypeName in reportTypes:
			print(*s)
	pr("getNodeData", nodeTypeName)
	mclass = om.MNodeClass(nodeTypeName)
	typeIdInt = None
	if(mclass.typeId.id() != 0):
		pr("legit typeIdInt found", mclass.typeId.id(), mclass.typeName)
		typeIdInt = mclass.typeId.id()

	# get node class bases
	baseClasses = cmds.nodeType(nodeTypeName, isTypeName=1, inherited=1) or []
	baseClasses = ["_BASE_"] + baseClasses[:-1]  # remove the last one, which is the node type itself

	# get type ids for all attributes
	try:
		attrs : list[om.MObject] = mclass.getAttributes()
	except:
		# abstract and intermediate node classes may not define any attributes
		# especially for MPxNode parents
		#print("failed to get attributes for", nodeTypeName)
		attrs = []
	# get all attribute data
	attrDatas = list(sorted((getAttrData(attr) for attr in attrs), key=lambda x: x.name))

	# apiTypeConstant = 4
	# apiTypeStr = "kDependencyNode"
	# mfnStr = "MFnDependencyNode"
	apiTypeConstant = ""
	apiTypeStr = ""
	mfnStr = ""
	try:
		pr("nodetype", nodeTypeName, )
		# mfn : om.MFnBase = getCache().nodeTypeLeafMFnMap[nodeTypeName](om.MObject())

		mfn: om.MFnBase = getCache().nodeTypeLeafMFnMap[nodeTypeName]
		kStr = getCache().nodeTypeNameToKStr(nodeTypeName)
		apiTypeConstant = getCache().classNameConstantMaps[om.MFn][kStr]
		apiTypeStr = getCache().classConstantNameMaps[om.MFn][apiTypeConstant]

		mfnStr = mfn.__name__
	except Exception as e:
		#log("error gathering node:", nodeTypeName)
		#traceback.print_exc()
		if nodeTypeName in reportTypes:
			traceback.print_exc()

	data = NodeData(
		typeName=nodeTypeName,

		apiTypeConstant=apiTypeConstant,
		apiTypeStr=apiTypeStr,
		mfnStr=mfnStr,
		typeIdInt=typeIdInt,
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
		apiTypeStr="kBase",
		typeIdInt=-1,
		mfnStr="MFnDependencyNode",
		attrDatas=[ messageData, cachingData, frozenData,
		            isHistoricallyInterestingData, nodeStateData ],
		bases=(),
		isAbstract=True,
	)


def sortNodeTypesByNBases(nodeTypes:T.Iterable[str])->dict[int, set[str]]:
	"""sort node types by number of base classes
	returns a dict mapping number of bases to list of node types
	"""
	typeLenSetMap : dict[int, set[str]] = defaultdict(set)
	for nodeType in nodeTypes:
		# get node class bases
		baseClasses = cmds.nodeType(nodeType, isTypeName=1, inherited=1) or []
		typeLenSetMap[len(baseClasses)].add(nodeType)
	return typeLenSetMap




def _gatherNodeData(nodeTypes=None):
	"""gather node data from maya
	"""
	# get all node types
	nodeTypes = nodeTypes or cmds.allNodeTypes(includeAbstract=1)

	# this adds " (abstract)" to the end of abstract node types :)
	abstractMap = {x.replace(" (abstract)", ""): " (abstract)" in x for x in nodeTypes}
	nodeTypes = [x.replace(" (abstract)", "") for x in nodeTypes]

	# check that all base classes are included
	nodeTypeSet = set(nodeTypes)
	for i in nodeTypes:
		nodeTypeSet.update(cmds.nodeType(i, isTypeName=1, inherited=1) or ())

	typeLenSetMap = sortNodeTypesByNBases(nodeTypeSet)
	nodeData : dict[str, dict[int, NodeData] ]= {}

	for typeLen, typeSet in sorted(typeLenSetMap.items(), key=lambda x: x[0]):
		nodeData[str(typeLen)] = {}
		#for nodeType in sorted(typeSet, key=lambda x: x):
		for nodeType in sorted(typeSet):
			data = getNodeData(nodeType)
			data.isAbstract = abstractMap.get(nodeType, True)
			nodeData[str(typeLen)][nodeType] = data
			if nodeType in reportTypes:
				log("writing data", nodeType)
				log(data)
	nodeData[str(0)] = {"_BASE_" : getBaseNodeData() }
	return nodeData

def updateDataForNodes(nodeTypes:T.Iterable[str]=None,
                       outputPath:Path=TARGET_NODE_DATA_PATH
                       ):

	nodeData = _gatherNodeData(nodeTypes)
	if outputPath.exists():
		try:
			existData = orjson.loads(open(outputPath).read())
		except orjson.JSONDecodeError:
			existData = {}
	else:
		existData = {}
	for baseLen, data in nodeData.items():
		for nodeType, nodeLeafData in data.items():
			existData.setdefault(baseLen, {})[nodeType] = nodeLeafData

	# write to file
	with open(outputPath, "wb") as f:
		f.write(
			orjson.dumps(existData,
			             option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS
			             )
		)
	#log("gather done")
	return existData

"""
script:

from importlib import reload
from wpm.core.node._codegen import gather

reload(gather)

gather.gatherNodeData()


"""


# wow hikHandle is a bolshy node

