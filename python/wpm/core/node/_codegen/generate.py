
from __future__ import annotations
import typing as T

import json, sys, os, shutil, time
from pathlib import Path
from typing import TypedDict
from collections import defaultdict
from dataclasses import dataclass

import orjson

from wplib import wpstring, log

from wptree import Tree

#from wpm import WN, om, cmds
#from wpm.core import getMFn

from wplib.codegen.strtemplate import ClassTemplate, FunctionCallTemplate, FunctionTemplate, TextBlock, argT, argsKwargsT, Literal, Assign, Import, IfBlock, indent, Comment
#from wptool.codegen import CodeGenProject

"""
base classes for nodes, plugs and others

this gen script kept clean of maya calls so we can 
iterate faster from pycharm direct


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
	apiTypeStr : str
	typeIdInt : int
	mfnStr : str
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

class MObjectDescriptor:
	"""descriptor used to return an attribute-level MObject
	for a child attribute of a compound array,
	or internal - something not otherwise accessible without
	a plug"""
	def __init__(self, name:str):
		self.name = name



class NodeBase:
	"""base class for nodes"""

	def plug(self, name:str):
		"""return a plug tree instance for this node"""



DEFAULT_NODE_JSON_PATH = Path(__file__).parent / "nodeData.json"

templatePath = Path(__file__).parent/ "template.py"
templateStr = templatePath.read_text()

def refPathForNodeType(project:CodeGenProject, nodeType:str):
	return project.refPath / (nodeType + ".py")

def genPathForNodeType(nodeType:str):
	return genDir / (nodeType + ".py")

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

def genNodeFileStr(data:NodeData,
                   nodeTypeAttrSetMap:dict[str, set[str]])->str:
	"""generate whole file string for a single node

	PROBLEM - hinting "Transform" in the plugs points to the generated
	class in the same file, not the final (potential) overridden
	Transform in the authored file
	"""


	nodeType = data.typeName
	# import statements first
	importLines = []
	parent = None
	if data.bases: # get parent classes (gen at runtime, author at compile time)
		parent = data.bases[-1]
		# no direct import, go through retriever class
		importLines.append("from .. import retriever")
		# Dag = retriever.getNodeCls("Dag")
		importLines.append(f"{wpstring.cap(parent)} = retriever.getNodeCls(\"{wpstring.cap(parent)}\")")

		# type-checking time import for final user-authored file
		# for parent class and for node itself,
		# to set hint ".node" attribute on plugs
		typeCheckImports = IfBlock(
			conditionBlocks=[["T.TYPE_CHECKING",
			                  ["from .. import " + wpstring.cap(parent),
			                   "from .. import " + wpstring.cap(nodeType)],
			                   ]],
		)
		importLines.append(str(typeCheckImports))

	importBlock = "\n".join(importLines)

	# attribute datas
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
		className = wpstring.cap(attrData.name) + "Plug"

		parentName = ""
		if attrData.parentName:
			parentName = wpstring.cap(attrData.parentName) + "Plug"

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
			childName = wpstring.cap(childData.name) + "Plug"
			descriptorCall = FunctionCallTemplate(
				"PlugDescriptor",
				fnArgs=((Literal(childData.name), ), {} )
			)
			childLines.append(Assign((childData.name + "_", childName), descriptorCall))
			childLines.append(Assign((childData.shortName + "_", childName), descriptorCall))

		# set node attr
		childLines.append(Assign(("node", wpstring.cap(nodeType)),))

		classDef = ClassTemplate(
			className=className,
			classBaseClasses=("Plug",),
			classLines=childLines,
		)
		plugTemplates.append(classDef)

	plugAssigns = []
	for attrData in newAttrDatas:
		descriptorCall = FunctionCallTemplate(
					"PlugDescriptor",
					fnArgs=((Literal(attrData.name), ), {} )
				)
		plugAssigns.append(Assign((attrData.name + "_", wpstring.cap(attrData.name) + "Plug"), descriptorCall))

	# node attributes
	nodeAssigns = []
	nodeAssigns.append(Assign("typeName", Literal(nodeType)))
	#log("apiTypeConstant", data.apiTypeConstant)
	if str(data.apiTypeConstant).strip():
		nodeAssigns.append(Assign("apiTypeInt", data.apiTypeConstant))
		nodeAssigns.append(Assign("apiTypeStr", Literal(data.apiTypeStr)))
	# MFn type
	# TODO: get the correct MFn type for the node, gather from maya
	nodeAssigns.append(Assign("MFnCls", "om." + data.mfnStr))
	#nodeAssigns.append(Assign(("MFn", "om." + data.apiTypeStr)))


	# generate main node
	if parent is not None:
		classBaseClasses = (wpstring.cap(parent),)
	else:
		classBaseClasses = ("WN",)
	nodeDef = ClassTemplate(
		className=wpstring.cap(nodeType),
		# classBaseClasses=(string.cap(parent),),
		classBaseClasses=classBaseClasses,
		classLines=plugAssigns + [Comment(), Comment("node attributes"), Comment()] +  nodeAssigns,
	)


	plugDefStrings = "\n".join([str(i) for i in plugTemplates])

	# write out final file
	fileName = nodeType + ".py"
	fileContent = templateStr.format(
		IMPORT_BLOCK=importBlock,
		DOC_BLOCK="",
		ATTR_DEF_BLOCK=plugDefStrings,
		NODE_DEF_BLOCK=nodeDef)
	#(project.refPath / fileName).write_text(fileContent)
	return fileContent


def genNodes(#project:CodeGenProject
		genDir:Path = Path(__file__).parent.parent / "gen",
		onlyTransform=False,
		onlyBase=False,
		jsonPath=DEFAULT_NODE_JSON_PATH,
		refreshGenInitFile=True
             ):
	"""generate nodes from json data"""

	startTime = time.time()
	with open(jsonPath, "r") as f:
		nodeData = json.load(f)
	readTime = time.time() - startTime
	print("read time", readTime)
	if onlyTransform:
		# start with venerable transform
		targetNodes = [

		]

		# get all the bases of transform
		nameNBasesMap = {}
		for nBases, nameMap in nodeData.items():
			for name, data in nameMap.items():
				nameNBasesMap[name] = nBases

		bases = nodeData["4"]["transform"]["bases"]
		for base in bases:
			if base not in nameNBasesMap:
				continue
			targetNodes.append(nodeData[nameNBasesMap[base]][base])
		targetNodes.append(nodeData["4"]["transform"])

	elif onlyBase:
		targetNodes = []
		try:
			for nBases, nameMap in nodeData.items():
				for name, data in nameMap.items():
					targetNodes.append(data)
					raise RuntimeError
		except RuntimeError:
			pass

	else: # regenerate every single node in maya
		targetNodes = []
		for nBases, nameMap in nodeData.items():
			for name, data in nameMap.items():
				targetNodes.append(data)

	# map of { node type : all attr names in that node type }
	# to avoid duplication
	typeAttrSetMap : dict[str, set[str]] = {}

	genStartTime = time.time()
	for i in targetNodes:

		#print("generate node", i["typeName"], "bases", i["bases"])

		# recover node data from json
		nodeData = recoverNodeData(i)
		# generate node file to write
		nodeFileStr = genNodeFileStr(
			data=nodeData, nodeTypeAttrSetMap=typeAttrSetMap)

		# write to gen folder
		genPathForNodeType(nodeData.typeName).write_text(nodeFileStr)
		pass
	print("generated {} nodes in {}".format(
		len(targetNodes), time.time() - genStartTime) )

	if refreshGenInitFile:
		# first recreate main generated init file
		processGenInitFile(genDir / "__init__.py", genDir)

		# import gen catalogue to author dir init file
		processGenInitFile(
			authorDir / "__init__.py", authorDir,
			extraImports=[Import(fromModule="..gen", module="Catalogue", alias="GenCatalogue")],
			catalogueBases=("GenCatalogue",)
		)


def processGenInitFile(initFile:Path,
                       gatherDir:Path,
                       extraImports:list[Import]=(),
                       catalogueBases=(),
                       ):
	"""write out any extra imports in __init__.py files
	and populate assignments to Catalogue class -
	init files don't import any real classes to avoid cycles

	we now generate a type checking import block and a
	type-check-time alternate Catalogue class

	janky args for different behaviour in author vs gen
	"""
	initFileText = initFile.read_text()

	# get all imports
	imports = extraImports or []
	assignments = []
	for refFile in gatherDir.iterdir():
		if refFile.suffix != ".py":
			continue
		if refFile.stem == "__init__":
			continue
		imports.append(Import(
			fromModule="." + refFile.stem,
			module=wpstring.cap(refFile.stem),
		))
		assignments.append(Assign(
			wpstring.cap(refFile.stem), wpstring.cap(refFile.stem)
		))

	catalogueClassDef = ClassTemplate(
		className="Catalogue",
		#classBaseClasses=("T.Protocol",),
		classBaseClasses=catalogueBases,
		classLines=assignments,
	)

	typeCondition = IfBlock(
		[["T.TYPE_CHECKING", imports + [catalogueClassDef]]]
	)


	initFileText = initFileText.format(
		TYPE_CHECK_BLOCK=typeCondition)

	initFile.write_text(initFileText)


genDir = Path(__file__).parent.parent / "gen"
genInitTemplatePath = Path(__file__).parent / "__genInit__.py"
authorDir = Path(__file__).parent.parent / "author"
authorInitTemplatePath = Path(__file__).parent / "__authorInit__.py"

def resetGenDir():

	# first clear out old stuff
	shutil.rmtree(genDir, ignore_errors=True)
	genDir.mkdir()

	# copy over init files
	shutil.copy2(genInitTemplatePath, genDir / "__init__.py")
	shutil.copy2(authorInitTemplatePath, authorDir / "__init__.py")



if __name__ == '__main__':

	print(genDir, genDir.is_dir())

	resetGenDir()

	genNodes(
		onlyTransform=False,
		onlyBase=False,

		refreshGenInitFile=True
	)






