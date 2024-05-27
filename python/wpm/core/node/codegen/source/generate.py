
from __future__ import annotations
import typing as T

import json, sys, os
from pathlib import Path
from typing import TypedDict
from collections import defaultdict
from dataclasses import dataclass

import orjson

from wplib import string

from wptree import Tree

#from wpm import WN, om, cmds
#from wpm.core import getMFn

from wplib.codegen.strtemplate import ClassTemplate, FunctionCallTemplate, FunctionTemplate, TextBlock, argT, argsKwargsT, Literal, Assign, Import, IfBlock, indent
from wptool.codegen import CodeGenProject

"""
base classes for nodes, plugs and others

test using mixin bases to denote inputs, formats etc
to let ide catch errors

every attribute of every node gets its own class - 
loaded lazily

node.tx_
node.tx_() -> float
node.t_() -> list[float] # or mvector?


"""


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
	childDatas:tuple[AttrData] = ()
	parentData:tuple=()



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

def recoverNodeData(baseData:dict):

	attrDatas = []
	for attrDict in baseData["attrDatas"]:
		attrDatas.append(AttrData(**attrDict))
	baseData.pop("attrDatas")
	nodeData = NodeData(**baseData, attrDatas=attrDatas)
	return nodeData


class AttrDescriptor:
	"""intermediate object to separate attribute
	namespace from main node functions -
	"""

class PlugDescriptor:
	"""descriptor for plugs -
	declare whole attr hierarchy in one go"""
	def __init__(self, name:str, mfnType:type[om.MFnAttribute]):
		self.name = name
		self.mfnType = mfnType

	# TEMP get and set
	def __get__(self, instance:NodeBase, owner):
		return instance.plug(self.name)
	# TEMP
	# def __set__(self, instance, value):
	# 	self.value = value


class NodeBase:
	"""base class for nodes"""

	def plug(self, name:str):
		"""return a plug tree instance for this node"""



jsonPath = Path(__file__).parent / "nodeData.json"

templatePath = Path(__file__).parent/ "template.py"
templateStr = templatePath.read_text()

def refPathForNodeType(project:CodeGenProject, nodeType:str):
	return project.refPath / (nodeType + ".py")

def genPathForNodeType(project:CodeGenProject, nodeType:str):
	return project.genPath / (nodeType + ".py")

def modifiedPathForNodeType(project:CodeGenProject, nodeType:str):
	return project.modifiedPath / (nodeType + ".py")


def sortAttrDatas(datas:tuple[AttrData]):
	# sort attributes such that leaf attrs come first
	nameDataMap :dict[str, AttrData]= {d.name : d for d in datas}
	attrDatas = list(sorted(datas, key=lambda x:x.name))
	# set child and parent references
	for d in attrDatas:
		if d.parentName:
			d.parentData = nameDataMap.get(d.parentName)
			nameDataMap[d.parentName].childDatas += (d, )
	# reorder
	for d in tuple(attrDatas):
		if d.parentName:
			attrDatas.remove(d)
			attrDatas.insert(attrDatas.index(d.parentData), d)
	return attrDatas

def genNodeFileStr(project:CodeGenProject, data:NodeData,
                   nodeTypeAttrSetMap:dict[str, set[str]])->str:
	#print("gen node", data)
	nodeType = data.typeName
	# import statements first
	importLines = []
	parent = data.bases[-1]
	importLines.append("from .{} import {}".format(parent, string.cap(parent)))
	importBlock = "\n".join(importLines)

	attrDatas = sortAttrDatas(data.attrDatas)
	nodeTypeAttrSetMap[data.typeName] = {i.name for i in attrDatas}
	# look up combined set of preceding base attributes
	attrSet = set()
	for baseName in data.bases:
		attrSet.update(nodeTypeAttrSetMap.get(baseName, set()))
	newAttrDatas = [i for i in attrDatas if not i.name in attrSet]

	# maybe we don't need a separate class for every kind of plug -
	# consider defining generic, with a specific namedTuple for each one,
	# not a separate subclass
	# define new plug descriptors for attributes # why though

	plugTemplates = []
	#for attrData in attrDatas:
	for attrData in newAttrDatas:
		className = string.cap(attrData.name) + "Plug"

		parentName = ""
		if attrData.parentName:
			parentName = string.cap(attrData.parentName) + "Plug"

		#childMap : dict[argT, FunctionCallTemplate] = {}
		childLines = []
		if parentName: # descriptor for parent plug
			childLines.append(
				Assign(("parent", parentName),
					FunctionCallTemplate(
						"PlugDescriptor",
						fnArgs=((Literal(attrData.parentName), ), {} )
					)
				)
			)
		for childData in attrData.childDatas:
			childName = string.cap(childData.name) + "Plug"
			descriptorCall = FunctionCallTemplate(
				"PlugDescriptor",
				fnArgs=((Literal(childData.name), ), {} )
			)
			childLines.append(Assign((childData.name + "_", childName), descriptorCall))
			childLines.append(Assign((childData.shortName + "_", childName), descriptorCall))

		# set node attr
		childLines.append(Assign(("node", string.cap(nodeType)),))

		classDef = ClassTemplate(
			className=className,
			classBaseClasses=("Plug",),
			classLines=childLines,
		)
		plugTemplates.append(classDef)

	plugAssigns = []
	for data in newAttrDatas:
		descriptorCall = FunctionCallTemplate(
					"PlugDescriptor",
					fnArgs=((Literal(data.name), ), {} )
				)
		plugAssigns.append(Assign((data.name + "_", string.cap(data.name) + "Plug"), descriptorCall))

	# generate main node
	nodeDef = ClassTemplate(
		className=string.cap(nodeType),
		classBaseClasses=(string.cap(parent),),
		classLines=plugAssigns,
	)


	plugDefStrings = "\n\n".join([str(i) for i in plugTemplates])

	# write out final file
	fileName = nodeType + ".py"
	fileContent = templateStr.format(
		IMPORT_BLOCK=importBlock,
		DOC_BLOCK="",
		ATTR_DEF_BLOCK=plugDefStrings,
		NODE_DEF_BLOCK=nodeDef)
	#(project.refPath / fileName).write_text(fileContent)
	return fileContent


def genNodes(project:CodeGenProject):
	"""generate nodes from json data"""
	with open(jsonPath, "r") as f:
		nodeData = json.load(f)

	# start with venerable transform
	targetNodes = [
		nodeData["4"]["transform"]
	]

	# map of { node type : all attr names in that node type }
	# to avoid duplication
	typeAttrSetMap : dict[str, set[str]] = {}

	for i in targetNodes:

		# recover node data from json
		nodeData = recoverNodeData(i)
		# generate node file to write
		nodeFileStr = genNodeFileStr(
			project, data=nodeData, nodeTypeAttrSetMap=typeAttrSetMap)

		# write to ref folder
		refPathForNodeType(project, nodeData.typeName).write_text(nodeFileStr)
		# write to gen folder
		genPathForNodeType(project, nodeData.typeName).write_text(nodeFileStr)
		pass

def makeCatalogueClass(project:CodeGenProject,
                       topDir:Path)->ClassTemplate:
	"""return a class with an attribute defined for
	all node classes in the given directory.
	This only matters for static typing,
	at runtime WN won't actually inherit from these

	class Catalogue:
		Transform : Transform = Transform
	# bloody poetry

	"""
	attrAssigns = []
	for refFile in topDir.iterdir():
		if refFile.suffix != ".py":
			continue
		if refFile.stem == "__init__":
			continue
		attrAssigns.append(Assign((
			string.cap(refFile.stem), string.cap(refFile.stem)), string.cap(refFile.stem)))
	return ClassTemplate(
		className="Catalogue",
		classBaseClasses=(),
		classLines=attrAssigns
	)

def processInitFiles(project:CodeGenProject):
	"""write out any extra imports in __init__.py files

	noted by {IMPORT_BLOCK} in each
	"""
	for topPath in (project.modifiedPath, project.genPath):
		initFile = topPath / "__init__.py"
		initFileText = initFile.read_text()

		# get all imports
		imports = []
		for refFile in topPath.iterdir():
			if refFile.suffix != ".py":
				continue
			if refFile.stem == "__init__":
				continue
			imports.append(Import(
				fromModule=refFile.stem,
				module=string.cap(refFile.stem),
			))
		importBlock = "\n".join([str(i) for i in imports])
		# wrap with guard for TYPE_CHECKING
		importBlock = (str(IfBlock(
			[ (TextBlock("T.TYPE_CHECKING"), indent(TextBlock(importBlock)))]
		)))
		initFileText = initFileText.format(IMPORT_BLOCK=importBlock)

		initFileText += "\n\n" + str(IfBlock(
			[ (TextBlock("T.TYPE_CHECKING"), makeCatalogueClass(project, topPath))]
		))

		initFile.write_text(initFileText)


if __name__ == '__main__':
	nodeCodeProject = CodeGenProject(
		Path(__file__).parent.parent
	)

	print(nodeCodeProject.topPath)
	nodeCodeProject.regenerate()
	genNodes(nodeCodeProject)
	processInitFiles(nodeCodeProject)



