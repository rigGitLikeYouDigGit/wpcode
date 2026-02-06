from __future__ import annotations

import traceback
import types, typing as T
import pprint
from pathlib import Path

import copy, json, importlib
from collections import defaultdict
from typing import NamedTuple, TypedDict
from uuid import uuid4
from orjson import loads

from wplib import log

from deepdiff import DeepDiff, Delta

import hou

from . import types
importlib.reload(types)

from .types import NodeHeader, ParmNames, CachedFile, dumps, loads

_dbgDepth = 0
def dbg(fn):
	def wrapper(*args, **kwargs):
		global _dbgDepth
		_dbgDepth += 1
		try:
			print("|--" * _dbgDepth, fn.__name__, args, kwargs)
			return fn(*args, **kwargs)
		finally:
			_dbgDepth -= 1
			# globals()["print"] = lambda *a, dbgDepth=_dbgDepth, printFn=globals()["print"],**k : printFn(
			# 	"\t" *
			#                                            dbgDepth, *a, **k)
	return wrapper

"""extract node deltas then params - 
for each node ref, save name and uid to allow both means

save the node's version mode and number as parametres, so if a node changes to an explicit version, that's a param change

save header,
meta,
connections,
params,
param dialog

TODO: add wildcard expressions for connections. just a little bit as a treat


hda def files should be named

gridBase_v01_textHDA.json

I really can't explain the mental block I have with writing houdini python
code, I go in with the best intentions and it immediately turns to 
sweater spaghetti

for overriding parent attrs, provide 3 params for each - 
locked SUPER_attr , showing original value always, OVERRIDE_attr checkbox,
allowing local editing of the actual named attr


we use a push model, any node that changes will sync all active children defs
in the scene - each one saves its leaf AND TOTAL data in HDA sections, so 
pulling only needs to look at one level

"""

# FOR TESTING - supply list of paths to be searched for text HDA definitions
hdaDefDirs = [
	"C:/Users/arthu/Desktop/textHDAs/"
]

# def getDefNameVersion(name)->tuple[str, int]:
# 	tokens = name.split("_")
# 	return tokens[0], int("".join(i for i in tokens[1] if i.isdigit()))

NETWORK_BOX_S = "NETWORK_BOX"
TOP_SEP_CHAR = "@@"
VERSION_SEP_CHAR = "@"
TEXT_HDA_BUNDLE_NAME = "textHDA_bundle_nodes"

HDA_SECTIONS_TO_COPY = [
	"OnCreated",
	"PythonModule",
	#"OnDeleted",
	"PostLastDelete"
]

safeCharMap = {
	"\t" : "£TAB",
	"\"" : "£DQ",
	"\'" : "£Q"
}

class ParentBaseData(TypedDict):
	file : str
	text : str
	localOverride : str

def makeSafeForJson(s:str):
	for k, v in safeCharMap.items():
		s = s.replace(k, v)
	return s
def regenFromJson(s:str):
	for k, v in safeCharMap.items():
		s = s.replace(v, k)
	return s

def truncate2Places(f:float):
	return int(f * 100) / 100.0


def getNodeHeader(node: hou.Node, rootNode: hou.Node):
	path = rootNode.relativePathTo(node)
	if isinstance(node, hou.NetworkBox):
		return [path, "NETWORK_BOX", "", "", "", 0]
	typeInfo = node.type()
	"""
	Returns a tuple of node type name components that constitute the full node type name. The components in the tuple appear in the following order: scope network type, node type namespace, node type core name, and version.

	# parse the full name into components
	>>> node_type = hou.nodeType(hou.dopNodeTypeCategory(), 'pyrosolver::2.0')
	>>> node_type.nameComponents()
	('', '', 'pyrosolver', '2.0')

	if a node type's version is not the latest, do we assume it matters?
	or if a node type is not the exact one you get, from an inexact lookup, then it matters.

	Danger to switch silently if someone else publishes a new version of a contained node while you're working on the hda
	"""
	scopeType, nodeTypeNS, nodeTypeName, exactVersion = typeInfo.nameComponents()

	# check if this node type is the same you get with a general lookup
	# defaultType = hou.NodeType(hou.sopNodeTypeCategory(), nodeTypeName )
	defaultType = hou.nodeType(hou.sopNodeTypeCategory(), nodeTypeName)
	exactVersionMatters = int(
		defaultType.nameComponents() != typeInfo.nameComponents())

	# save everything EXCEPT the version at node level
	return [path, scopeType, nodeTypeNS, nodeTypeName, exactVersion,
	        exactVersionMatters,
	        truncate2Places(node.position()[0]),
	        truncate2Places(node.position()[1])]


@dbg
def createNodeFromHeader(rootNode: hou.Node, header: NodeHeader):
	"""create a new node from the given header: path and node type.

	Looks like we need to know the version (and if it matters) at node creation,
	so pack that on the end of this header?"""
	nodePath, scopeType, nodeTypeNS, nodeTypeName, exactVersion, exactVersionMatters, x, y = header
	nodeName = nodePath.split("/")[-1]
	if "/" in nodePath:
		parentNodePath = "/".join(nodePath.split("/")[:-1])
	else:
		parentNodePath = "."
	parentNode: hou.Node = rootNode.node(parentNodePath)

	if exactVersionMatters:
		exactVersion = "::".join(
			filter(None, (nodeTypeNS, nodeTypeName, exactVersion)))
		print("CREATE EXACT VERSION ", exactVersion, nodeName)
		newNode = parentNode.createNode(
			exactVersion, nodeName, exact_type_name=True
		)
	else:
		newNode = parentNode.createNode(
			nodeTypeName, nodeName
		)
	newNode.setPosition(hou.Vector2(x, y))
	return newNode


def getChildrenConnectionData(parentNode: hou.OpNode) -> list[list[str]]:
	"""very simple, of form
	[ 0-myNodeOutputName-myNode , 0-myNodeInputName-myOtherNode ]"""
	connections = []
	print("getChildrenConnectionData", parentNode, parentNode.children())
	for node in parentNode.children():
		node: hou.OpNode
		connectors: tuple[tuple[hou.NodeConnection]] = node.inputConnectors()
		for i, connector in enumerate(connectors):
			if not connector:  # input not driven
				continue
			if subnetInput := connector[0].subnetIndirectInput():
				connections.append(
					[f"{subnetInput.number()}",
					 f"{i}-{connector[0].outputName()}-{node.name()}"]
				)
			else:
				connections.append(
					[f"{connector[0].inputIndex()}-{connector[0].inputName()}-{connector[0].inputNode().name()}",
					 f"{i}-{connector[0].outputName()}-{node.name()}"]
				)
	return connections


def setChildrenConnectionData(
		parentNode: hou.Node, connections: list[list[str]],
		useNames=True
):
	"""very simple, of form
	[ 0-myNodeOutputName-myNode , 0-myNodeInputName-myOtherNode ]

	TODO: add error checks throughout this to fall back if using names fails
	"""
	subnetInputs = parentNode.indirectInputs()
	for c in connections:
		dstIndex, dstInName, dstName = c[1].split("-")
		dstNode: hou.Node = parentNode.node(dstName)
		if len(c[0].split("-")) == 1:  # subnet input
			srcItem = subnetInputs[int(c[0])]
			if useNames:
				dstNode.setNamedInput(dstInName, srcItem, 0)
			else:
				dstNode.setInput(int(dstIndex), srcItem, 0)
			continue

		srcIndex, srcInName, srcName = c[0].split("-")
		srcNode = parentNode.node(srcName)
		if useNames:
			dstNode.setNamedInput(dstInName, srcNode, srcInName)
		else:
			dstNode.setInput(int(dstIndex), srcNode, int(srcIndex))


def getConnectionsDiff(
		baseData: list[list[str]],
		newData: set[NodeHeader]
) -> dict[str, list[list[str]]]:
	"""return a dict of {"add" : [], "del" : [] }
	"""
	result = {"add": [], "del": []}
	for i in baseData:
		if not i in newData:
			result["del"].append(i)
	for i in newData:
		if not i in baseData:
			result["add"].append(i)
	return result


def getNodeParamText(node: hou.OpNode, saveDefaults=True) -> dict[str, T.Any]:
	"""get verbose so we can iterate over dictionary easier when diffing
	"""
	result = {}
	for i in node.parms():
		i: hou.Parm
		if not saveDefaults:
			if i.isAtDefault():
				continue
		result[i.name()] = i.asData()
	return result


def setNodeParamsFromText(node: hou.OpNode, data):
	for parmName, v in data.items():
		node.parm(parmName).setFromData(v)


def getNodeParmValues(node: hou.Node) -> dict[str, T.Any]:
	""""""
	return {
		i.name(): i.eval() for i in node.parms()
	}


def setNodeParmValues(node: hou.Node, data: dict):
	for k, v in data.items():
		if not node.parm(k):
			continue
		node.parm(k).set(v)


"""very quick and dirty deep diff and update system"""


def deepUpdatePath(baseData: dict | list, path: list, value):
	token, path = path[0], path[1:]
	if isinstance(baseData, dict):
		if not path:
			baseData[token] = value
			return
		if not token in baseData:
			baseData[token] = value
			return
		deepUpdatePath(baseData, path, value)

	elif isinstance(baseData, list):
		index = -1
		if isdigit(token):
			index = int(token)
		elif isinstance(token, int):
			index = token
		else:
			print(baseData)
			print(token, path)
			raise RuntimeError("invalid list index:", token)
		if index >= len(baseData):
			baseData.append(value)
			return
		if not path:
			baseData[index] = value
			return
		return deepUpdatePath(baseData[index], path, value)


"""generate separate paths from nested patch dict"""


def deepUpdate(baseData: dict | list, patch: list[tuple[list[str], T.Any]]):
	for k, v in patch:
		deepUpdatePath(baseData, list(k), v)


def deepDiffParams(
		baseData,
		newData
) -> Delta:
	diff = DeepDiff(baseData, newData)
	return Delta(diff)


def applyDeepPatchParams(
		baseData,
		patch: Delta
):
	return baseData + patch


def diffParamsText(
		baseData: dict,
		newData: dict,
		patchData: dict = None
):
	""" better to be diffed
	than to be no-diffed
	"""
	if patchData is None:
		patchData = {}

	# if isinstance(newData, (tuple, list)):
	# 	if not isinstance(baseData, (tuple, list)):
	# 		return newData
	# 	for i,
	for k, v in newData.items():
		if not k in baseData:
			patchData[k] = diffParamsText({}, newData)
			# patchData[k] = {}
			# diffParamsText({}, v, patchData[k])
			continue


def getParmDialogScripts(node: hou.Node) -> dict[str]:
	"""trying to save parm definitions in dialog script to ensure we
	keep everything -
	for some reason you can't get that for individual templates.

	So for each parm, make a new template group containing only it,
	and get that as dialog
	"""
	result = {}
	for p in node.spareParms():
		p: hou.Parm
		template = p.parmTemplate()
		try:
			"""sometimes fails with 'sequence of parm templates cannot 
			include FolderSetParmTemplates', whole system is insane"""
			ptg = hou.ParmTemplateGroup([template])
		except hou.OperationFailed:
			continue
		result[p.name()] = makeSafeForJson(ptg.asDialogScript())
	return result


@dbg
def getTextHDAParmDialogScripts(node: hou.Node):
	"""special-case textHDA root nodes - maybe this isn't
	necessary, but I don't want the system accidentally
	erasing itself
	"""
	hda = TextHDANode(node)
	print("leaf pts", hda.leafHDAParmTemplates())
	# leafPtg = hou.ParmTemplateGroup(hda.leafHDAParmTemplates())
	# parentPtg = hou.ParmTemplateGroup(hda.parentHDAParmTemplates())
	result = {
		# "parent" : {i.name() : makeSafeForJson(hou.ParmTemplateGroup([
		# 	i]).asDialogScript())
		#             for i in hda.parentHDAParmTemplates()},
		"LEAF": {i.name(): makeSafeForJson(hou.ParmTemplateGroup([
			i]).asDialogScript())
		         for i in hda.leafHDAParmTemplates()},
	}
	print("texthda dialog scripts:")
	print(result)
	return result

def copyFolderPT(folderPT: hou.FolderParmTemplate) -> hou.FolderParmTemplate:
	return hou.FolderParmTemplate(
		folderPT.name(), folderPT.label(),
		(),
		folderPT.folderType(),
		tags=folderPT.tags(),
		conditionals=folderPT.conditionals(),
		tab_conditionals=folderPT.tabConditionals()
	)

@dbg
def setTextHDAParmDialogScripts(node: hou.OpNode, data: dict):
	"""remove existing hda parms and reset from given data
	we modify the current hda definition ptg used by the node.

	also handle setting params under parent vs leaf folders

	data will be dict of {
		"myHDaName" : {
			"parmName" : "dialogScript",
			}
		}
	"""
	print("text parms")
	pprint.pprint(data)
	hda = TextHDANode(node)
	defName = hda.defFile()
	ptg: hou.ParmTemplateGroup = hda.hdaDef().parmTemplateGroup()

	leafFolderPT = hda.leafHDAParmFolderTemplate()
	parentFolderPT = hda.parentHDAParmFolderTemplate()

	if not defName:
		if leafFolderPT:
			newLeafFolderPT = copyFolderPT(leafFolderPT)
			ptg.replace(defName, newLeafFolderPT)
			# for i in leafFolderPT.parmTemplates():
			# 	ptg.remove(i)

	newParentFolderPT = copyFolderPT(parentFolderPT)
	ptg.replace(parentFolderPT.name(), newParentFolderPT)
	# for i in parentFolderPT.parmTemplates():
	# 	if isinstance(i, hou.FolderParmTemplate):
	# 		newParentFolderPT = copyFolderPT(i)
	# 		ptg.appendToFolder(parent)
	# 	ptg.remove(i)

	# check any duplicate names still in ptg
	print("AFTER CLEAR PTG:")
	for i in ptg.parmTemplates():
		if isinstance(i, hou.FolderParmTemplate):
			for i in i.parmTemplates():
				if isinstance(i, hou.FolderParmTemplate):
					for i in i.parmTemplates():
						print(i.name(), i.label())
					continue
				print(i.name(), i.label())
			continue
		print(i.name(), i.label())

	for hdaDefName, paramData in data.items():
		# remove leaf section if no def name

		# assign same-node params to leaf
		if hdaDefName == defName:
			leafFolderPT = hda.leafHDAParmFolderTemplate()
			for k, v in paramData.items():
				parmPTG = hou.ParmTemplateGroup()
				parmPTG.setToDialogScript(regenFromJson(v))
				# pt : hou.ParmTemplate = parmPTG.parmTemplates()[0]
				pt: hou.ParmTemplate = parmPTG.find(k)
				ptg.appendToFolder(ParmNames.leafHDAParmFolderLABEL, pt)
				#leafFolderPT.addParmTemplate(pt)
			# if ptg.findFolder(ParmNames.leafHDAParmFolderLABEL):
			# 	ptg.replace(ParmNames.leafHDAParmFolder, leafFolderPT)
			# else:
			# 	ptg.append(leafFolderPT)
			continue
		# assign other-node params to parent
		# first create top-level folders under main parent, then append to them?

		defFolderPT = ptg.findFolder(hdaDefName)
		if not defFolderPT:
			defFolderPT = hou.FolderParmTemplate(
				hdaDefName, hdaDefName, ())
			ptg.appendToFolder(parentFolderPT, defFolderPT)
		for i in defFolderPT.parmTemplates():
			ptg.remove(i)

		pts = []
		for k, v in paramData.items():
			parmPTG = hou.ParmTemplateGroup()
			parmPTG.setToDialogScript(regenFromJson(v))
			# pt : hou.ParmTemplate = parmPTG.parmTemplates()[0]
			pt: hou.ParmTemplate = parmPTG.find(k)
			ptg.appendToFolder(defFolderPT, pt)
			#pts.append(pt)
		#
		# parentFolderPT.addParmTemplate(defFolderPT)
		#
		# if ptg.findFolder(defFolderPT.label()):
		# 	ptg.remove(hdaDefName.label())
	#ptg.replace(parentFolderPT.name(), parentFolderPT)
	print()
	print("BEFORE SET PTG")
	for i in ptg.parmTemplates():
		if isinstance(i, hou.FolderParmTemplate):
			for i in i.parmTemplates():
				if isinstance(i, hou.FolderParmTemplate):
					for i in i.parmTemplates():
						print(i.name(), i.label())
					continue
				print(i.name(), i.label())
			continue
		print(i.name(), i.label())
	hda.hdaDef().setParmTemplateGroup(
		ptg, rename_conflicting_parms=False, create_backup=False)
	return
	parmNames = {"PARENT": (ParmNames.parentHDAParmFolderLABEL,
	                        hou.ParmTemplateGroup()),
	             "LEAF": (ParmNames.leafHDAParmFolderLABEL,
	                      hou.ParmTemplateGroup())
	             }

	for cat in ("PARENT", "LEAF"):
		folderPT: hou.FolderParmTemplate = ptg.findFolder(parmNames[cat][0])

		# print("folderPT", folderPT)
		ptg.replace(folderPT, folderPT)
		origFolderPT: hou.FolderParmTemplate = ptg.findFolder(parmNames[cat][0])
		folderPT.setParmTemplates(())
		ptg.replace(origFolderPT, folderPT)
		folderPT: hou.FolderParmTemplate = ptg.findFolder(parmNames[cat][0])
		origFolderPT: hou.FolderParmTemplate = ptg.findFolder(parmNames[cat][0])
		for k, v in data.get(cat, {}).items():
			parmPTG = hou.ParmTemplateGroup()
			parmPTG.setToDialogScript(regenFromJson(v))
			# pt : hou.ParmTemplate = parmPTG.parmTemplates()[0]
			pt: hou.ParmTemplate = parmPTG.find(k)

			folderPT.addParmTemplate(pt)

		ptg.replace(origFolderPT, folderPT)

	# update new folders on node
	hda.hdaDef().setParmTemplateGroup(
		ptg, rename_conflicting_parms=False, create_backup=False)


def setParmDialogScripts(node: hou.Node, ptgData: dict):
	masterPtg: hou.ParmTemplateGroup = node.parmTemplateGroup()
	for parmName, data in ptgData.items():
		ptg = hou.ParmTemplateGroup()
		ptg.setToDialogScript(regenFromJson(data))
		masterPtg.append(ptg.parmTemplates()[0])
	node.setParmTemplateGroup(masterPtg)


def iterNodesToTrack(topNode: hou.Node) -> list[hou.Node]:
	"""does not return top node"""
	result = []
	toIter = list(topNode.children())
	while toIter:
		node = toIter.pop(-1)
		result.append(node)
		if isTextHDANode(node):
			continue
		if node.isInsideLockedHDA():
			if node.isEditableInsideLockedHDA():
				toIter.append(node)
				continue
		elif node.isLockedHDA():
			continue
		toIter.extend(node.children())
	return result


paramsToIgnore = list(ParmNames.__dict__.values())


@dbg
def getFullNodeState(
		node: hou.OpNode
) -> dict:
	"""get full snapshot of node -
	prune at included text hda nodes.

	top-level parms saved by hda def name
	"""
	hda = TextHDANode(node)
	nodes = iterNodesToTrack(node)
	print("nodes", nodes)
	# get params to add to the top node
	baseParmTemplateGroup: hou.ParmTemplateGroup = node.parmTemplateGroup()
	print("base ptg", baseParmTemplateGroup)

	parmVals = {}
	for i in nodes:
		parmValData = getNodeParamText(i)
		if not parmValData:
			continue
		parmVals[node.relativePathTo(i)] = parmValData

	parmTemplatesToAdd = {}
	for i in nodes:
		if isTextHDANode(i):
			# don't support adding spare inputs to textHDAs, we
			# have enough going on
			continue
		else:
			parmDialogData = getParmDialogScripts(i)
		if not parmDialogData:
			continue
		parmTemplatesToAdd[node.relativePathTo(i)] = parmDialogData

	# get top-level parm dialogs
	parmTemplatesToAdd["."] = hda.inheritedParmDialogScriptMap()

	# node paths are exclusive, so let those override
	result = {
		"nodes": {node.relativePathTo(i): getNodeHeader(i, node) for i in
		          nodes},
		"parmTemplates": parmTemplatesToAdd,
		# "connections" : [getChildrenConnectionData(i) for i in [node] + nodes],
		"connections": getChildrenConnectionData(node),
		# "parmVals" : {node.relativePathTo(i) : getNodeParamText(i) for i in nodes}
		"parmVals": parmVals
	}
	print("result:")
	pprint.pprint(result, width=2000)

	return result


"""we don't need sophisticated diffs for nodes, parm templates etc - 
overriding and adding should be fine, since dangling nodes
in the network LIKELY won't matter?

make that an option

"""

@dbg
def mergeNodeStates(
		baseData: dict,
		states: list[dict]
):
	log("mergeNodeStates:",)
	pprint.pprint(baseData)
	pprint.pprint(states)
	baseData.setdefault("nodes", {})
	baseData.setdefault("parmTemplates", {})
	baseData.setdefault("connections", [])
	baseData.setdefault("parmVals", {})

	for state in states:

		baseData["nodes"].update(state.get("nodes", {}))
		for k, v in state.get("parmTemplates", {}).items():
			baseData.setdefault(k, {})
			baseData[k].update(v)
		baseData["parmTemplates"].update(state.get("parmTemplates", {}))
		baseData["connections"].extend(state.get("connections", []))
		for nodePath, parmData in state.get("parmVals", {}).items():
			if not nodePath in baseData["parmVals"]:
				baseData["parmVals"][nodePath] = parmData
				continue
			baseData["parmVals"][nodePath].update(parmData)

	print("result:")
	pprint.pprint(baseData)
	return baseData


def parmValsAreEqual(a, b):
	"""dumb and stupid for now, make this more complex if needed
	"""
	return str(a) == str(b)


def diffNodeState(
		baseData: dict,
		wholeNodeState: dict
) -> dict:
	"""get final dict to put in leaf parm
	works only as an overkay/override
	"""
	result = {
		"nodes": {},
		"parmTemplates": {},
		"connections": [],
		"parmVals": {}
	}
	# ensure baseData has all the needed keys set up
	for k, v in result.items():
		baseData.setdefault(k, v)
	print("diffNodeState:")

	print(baseData["nodes"])
	print(wholeNodeState["nodes"])
	# print("base:")
	# pprint.pprint(baseData)
	# print("whole state:")
	# pprint.pprint(wholeNodeState)

	for nodePath, nodeData in wholeNodeState.get("nodes", {}).items():
		"""override node creation if leaf node is not found,
		or if it has a different type/version compared to base"""
		if not nodePath in baseData["nodes"]:
			result["nodes"][nodePath] = nodeData
			continue
		# check that version and name are identical
		if not all(nodeV == baseV
		           for nodeV, baseV in zip(
			nodeData[:6], baseData["nodes"][nodePath][:6])):
			result["nodes"][nodePath] = nodeData

	print("diff nodes:")
	print(result["nodes"].keys())
	for nodePath, parmData in wholeNodeState.get("parmTemplates", {}).items():
		if not nodePath in baseData["parmTemplates"]:
			result["parmTemplates"][nodePath] = parmData
			continue
		for parmName, parmScript in parmData.items():
			if str(parmScript) != str(
					baseData["parmTemplates"][nodePath].get(parmName, "")):
				result["parmTemplates"].setdefault(nodePath, {})
				result["parmTemplates"][nodePath][parmName] = parmScript

	print("connections")
	print(baseData["connections"])
	print(wholeNodeState["connections"])
	comp = set(map(tuple, baseData.get("connections", [])))
	print("comp", comp)
	for i in wholeNodeState.get("connections", []):
		if not tuple(i) in comp:
			result["connections"].append(i)

	for nodePath, parmData in wholeNodeState.get("parmVals", {}).items():
		if not nodePath in baseData["parmVals"]:  # new node
			result["parmVals"][nodePath] = parmData
			continue
		for parmName, parmVal in parmData.items():
			if not parmName in baseData["parmVals"][nodePath]:  # new parm
				result["parmVals"].setdefault(nodePath, {})
				result["parmVals"][nodePath][parmName] = parmVal
				continue
			if not parmValsAreEqual(
					baseData["parmVals"][nodePath][parmName],
					parmVal):  # save if not equal
				result["parmVals"].setdefault(nodePath, {})
				result["parmVals"][nodePath][parmName] = parmVal

	return result



def getNameVersionFromFileName(s:str)->(str, int):
	"""for a file named
	gridTest_v02.json
	return (gridTest, 2)
	"""
	stem, *suffix = s.split(".")
	tokens = stem.split("_")
	endTokenIndex = 0
	version = -1
	for i, tok in enumerate(tokens):
		# check if this token is version - if yes, all previous are the name
		versionTest = tok.replace("v", "").replace("V", "")
		if versionTest.isdigit():
			endTokenIndex = i
			version = int(versionTest)
			break
		if tok == "textHDA":
			endTokenIndex = i

	if endTokenIndex == 0: # no name found
		return "", -1
	return "_".join(tokens[:endTokenIndex+1]), version

def hdaEmbeddedIdDefMap()->dict[str, hou.HDADefinition]:
	result = {}
	for i in hou.hda.definitionsInFile("Embedded"):
		i : hou.HDADefinition
		result[i.nodeTypeName()] = i
		result[i.nodeTypeName().lower()] = i
	return result

def getFileHDADefs()->dict[str, dict[int, Path]]:
	results = defaultdict(dict)
	for hdaDir in hdaDefDirs:
		dirPath = Path(hdaDir)
		if not dirPath.is_dir():
			continue
		dirHdas = dirPath.glob("*_textHDA.json")
		for i in dirHdas:
			name, version = getNameVersionFromFileName(i.stem)
			if not name:
				continue
			results[name][version] = i
	return results

def getSceneHDADefNodes()->dict[str, list[hou.Node]]:
	result = defaultdict(list)
	nodes = allSceneTextHDANodes()
	print("nodes", nodes)

	for node in allSceneTextHDANodes():
		hdaNode = TextHDANode(node)
		if not hdaNode.defFileParm():
			continue
		# skip if it points to a real file path on disk
		if isinstance(hdaNode.defFile(), Path):
			continue
		result[hdaNode.defFile()].append(node)
	return result


def getHDADefFromPath(path:Path|str)->hou.HDADefinition|None:
	"""retrieve either a file or scene HDA def"""
	if Path(path).is_file():
		return hou.hda.definitionsInFile(str(path))[0]
	if result := hdaEmbeddedIdDefMap().get(path):
		return result
	return None

HDA_DELTA_SECTION_NAME = "textHDALeafDelta"
HDA_FULL_SECTION_NAME = "textHDAFullState"

def getHDASectionDict(hdaDef:hou.HDADefinition, sectionName:str, default={}):
	"""return stored section in hda as json dict"""
	log("getHDASectionDict", hdaDef, sectionName, hdaDef.sections().keys())
	if not hdaDef.hasSection(sectionName):
		hdaDef.addSection(sectionName)
		hdaDef.sections()[sectionName].setContents(dumps(default))
	return loads(hdaDef.sections()[sectionName].contents())

def setHDASectionDict(hdaDef:hou.HDADefinition, sectionName:str, data:dict):
	if not hdaDef.hasSection(sectionName):
		hdaDef.addSection(sectionName)
	hdaDef.sections()[sectionName].setContents(dumps(data))

def getDefParentLeafData(
		hdaDef: hou.HDADefinition
) -> tuple[dict, dict]:
	"""if hda has parent sections, return them as dicts,
	otherwise run the full retrieval.
	we assume that node and def data are equivalent
	"""



# def getNodeParentLeafData(
# 		node: hou.OpNode
# ) -> tuple[dict, dict]:
# 	"""if node has parent sections, return them as dicts,
# 	otherwise run the full retrieval
# 	"""
# 	hda = TextHDANode(node)
# 	for i in hda.parentDefPaths():
# 		parentNode =


@dbg
def isTextHDANode(node:hou.Node):
	"""return true if node is a textHDA root node"""
	print("name components", node.type().nameComponents())
	return any(i in node.type().nameComponents()[2].lower()
	           for i in ("texthda", ))

@dbg
def setNodeToState(
		node:hou.OpNode,
		state:dict
):
	"""conform node exactly to the given state - this WILL
	delete all non-tracked nodes and connections

	TODO: for this not to explode, we need to remove callbacks during editing
		BUT to add the callback again, we need to refer to the top-level
		nodefn function
		I'm too dumb for this man
	"""
	# sorry
	from .nodefn import (addNodeInternalCallbacks,
	                     removeNodeInternalCallbacks,
	                     removeHDAParamCallbacks, addHDAParamCallbacks)
	# sorry

	print("set node to state", node)
	wasEditable = node.isEditable()
	if wasEditable:
		#removeNodeInternalCallbacks(node)
		removeHDAParamCallbacks(node)
	else:
		node.allowEditingOfContents(False)
	node.deleteItems(node.children())
	for nodePath, nodeData in state["nodes"].items():
		createNodeFromHeader(node, nodeData)
	for nodePath, ptgData in state["parmTemplates"].items():
		setNode = node.node(nodePath)
		#if isTextHDANode(setNode):
		if nodePath == ".":
			setTextHDAParmDialogScripts(setNode, ptgData)
		else:
			setParmDialogScripts(node.node(nodePath), ptgData)
	setChildrenConnectionData(node, state["connections"])
	for nodePath, parmVals in state["parmVals"].items():
		assert node.node(nodePath), (f"Node path {nodePath} not already built "
		                             f"before setting parms")
		setNodeParamsFromText(node.node(nodePath), parmVals )
	path = node.path()
	node.type().definition().updateFromNode(node)
	node : hou.OpNode = hou.node(path)
	if wasEditable:
		# addNodeInternalCallbacks(node, inputRewired=False,
		#                          paramChanged=False, topNode=node)
		addHDAParamCallbacks(node)
	else:
		node.matchCurrentDefinition()

# namespace to find texthda nodes declared in file -
# one per name per active version
TEXT_HDA_NAMESPACE = "TEXTHDA"
SAVED_TO_SCENE_TOKEN = "Embedded" # hardcoded
TEXT_HDA_BASE_DEF_NAME = "textHDA"
def hdaIsSavedToScene(hdaDef:hou.HDADefinition):
	return hdaDef.libraryFilePath() == SAVED_TO_SCENE_TOKEN

def getRandomStringName()->str:
	"""would be cool to have a random string of actual words
	but for now uuid is fine"""


def getTextHDANodeBundle()->hou.NodeBundle:
	"""return a dumb node bundle to manually add nodes on creation,
	for more efficient tracking of textHDAs by HDA definition and
	def file"""
	if not hou.nodeBundle(TEXT_HDA_BUNDLE_NAME):
		hou.addNodeBundle(TEXT_HDA_BUNDLE_NAME)
	return hou.nodeBundle(TEXT_HDA_BUNDLE_NAME)

def allSceneTextHDANodes()->list[hou.OpNode]:
	return getTextHDANodeBundle().nodes()
@dbg
def getSceneTextHDANodesByDef()->dict[str|Path, list[hou.OpNode]]:
	result = {}
	for i in allSceneTextHDANodes():
		defStr = i.parm(ParmNames.defFile).evalAsString().strip()
		if "." in defStr and "/" in defStr:
			key = Path(defStr)
		else:
			key = defStr
		if not key in result:
			result[key] = []
		result[key].append(i)
	return result


@dbg
def getDefAffectedNodeMap()->dict[str|Path, list[hou.OpNode]]:
	result = defaultdict(list)
	for i in allSceneTextHDANodes():
		textHda = TextHDANode(i)
		affectingDefs = textHda.parentDefPaths()
		if not textHda.editingAllowed():
			if not textHda.defFile():
				continue
			affectingDefs += [textHda.defFile()]
		for parentDef in affectingDefs:
			if not parentDef:
				continue
			result[parentDef].append(i)
	return result



#_baseHDADef : hou.HDADefinition = None

def getBaseTextHDADef()->hou.HDADefinition:
	"""TODO: this only looks at one version,
	add something that defaults to """
	print("get base def:")
	found = hou.nodeType("Sop/textHDA::1.0")
	if found is not None:
		found = found.definition()
	return found
	# #global _baseHDADef
	# _baseHDADef = None
	# if _baseHDADef is None:
	#
	# #raise
	# return _baseHDADef

def getEmbeddedTextHDADefs()->list[tuple[hou.NodeType, hou.HDADefinition]]:
	results = []
	for name, nodeType in hou.sopNodeTypeCategory().nodeTypes().items():
		nodeType : hou.NodeType
		if (hdaDef := nodeType.definition()) is None: # only consider hdas
			continue
		if not hdaIsSavedToScene(hdaDef):
			continue
		# check if "TEXTHDA" in namespace
		scopeName, namespace, typeName, version = nodeType.nameComponents()
		if TEXT_HDA_NAMESPACE in namespace:
			results.append((nodeType, hdaDef))
	return results

"""
so we need a separate hda def for each sequence of parent defs,
plus a hda def for each node.
"""
@dbg
def deleteHDADefIfUnused(hdaDef:hou.HDADefinition):
	try:
		try:
			if not hdaDef.nodeType().instances():
				print("deleting unused hda:", hdaDef)
				hdaDef.destroy()
		except hou.OperationFailed:
			return
	except hou.ObjectWasDeleted: # already cleaned up
		pass

@dbg
def createLocalTextHDADefinition(
		node:hou.OpNode,
		newId,
		deleteAssignedHDA=True
)->tuple[hou.HDADefinition, hou.OpNode]:
	"""we just give each node its own hda whenever it's edited away
	from the raw textHDA baseline.

	in order to save the hdamodule on the new node hda type,
	we copy the original node and update the new definition
	from it before deleting it

	TODO: if I'm reading the docs right, there's a way to set HDA section
		contents paths to actual filenames? unsure if that would allow
		setting a literal python module as section, rather than
		relying on embedded code
	"""
	hdaNode = TextHDANode(node)
	baseTextHdaDef = getBaseTextHDADef()
	masterPTG = baseTextHdaDef.parmTemplateGroup()
	#tempNode = hou.copyNodesTo((node, ), node.parent())[0]
	#hdaNode.setHDADefId(newId)
	origParmData = getNodeParmValues(node)
	origName = node.name()
	#hdaNode.hdaDefParm().lock(False)

	# check if def already exists
	existNodeTypeName = f"Sop/{newId}"
	# rules of when to prepend Sop bewilder me
	existingType = hou.nodeType(existNodeTypeName)
	if existingType:
		newNode = node.changeNodeType(
			newId,
			keep_name=True,
			keep_parms=True,
			keep_network_contents=True,
			force_change_on_node_type_match=True
		)
		return existingType.definition(), newNode

	# delete the original def
	origType : hou.NodeType = node.type()
	origDef : hou.HDADefinition = origType.definition()

	# we have to change the actual node type to a subnetwork before making an
	# HDA?
	baseHMSections = [baseTextHdaDef.sections()[i] for i in HDA_SECTIONS_TO_COPY]
	# baseHMSection : hou.HDASection = baseTextHdaDef.sections()["PythonModule"]
	# baseOnCreatedSection : hou.HDASection = baseTextHdaDef.sections()[
	# 	"OnCreated"]
	node.setName(newId)
	nodePath = node.path()
	print(" before change to subnet")
	node = node.changeNodeType(
		"subnet", keep_name=True, keep_parms=True,
		keep_network_contents=True, force_change_on_node_type_match=True
	)
	print(" after change to subnet")

	# node = newNode
	hdaNode = TextHDANode(node)
	# need to re-get node from houdini object model
	node : hou.OpNode = hou.node(nodePath)
	print("new subnet node", node)
	# createDigitalAsset seems to always take the node's actual name
	newNode = node.createDigitalAsset(
		name=newId,
		change_node_type=True,
		create_backup=False,
		save_as_embedded=True
	)

	# and again
	node = hou.node(nodePath)
	print("new hda def node", node)

	newHDADef : hou.HDADefinition = newNode.type().definition()
	# 4 slots for inputs and outputs, should be fine for now
	newHDADef.setMinNumInputs(0)
	newHDADef.setMaxNumInputs(4)
	newHDADef.setMaxNumOutputs(4)

	for i in HDA_SECTIONS_TO_COPY:
		if not i in newHDADef.sections():
			section = newHDADef.addSection(i)
	for i, key in enumerate(HDA_SECTIONS_TO_COPY):
		newHDADef.setExtraFileOption(f"{key}/IsPython", True)
		newHDADef.sections()[key].setContents(baseHMSections[i].contents())

	newHDADef.setParmTemplateGroup(masterPTG, create_backup=False)
	print("set orig parm data on node:", node)
	print(" orig parm data", origParmData)
	setNodeParmValues(node, origParmData)
	newHDADef.save("Embedded", create_backup=False)

	if deleteAssignedHDA:
		if origDef.nodeTypeName() != baseTextHdaDef.nodeTypeName():
			deleteHDADefIfUnused(origDef)
	# restore base name
	newNode.setName(origName)
	print("completed type change")

	return newHDADef, node

def defIsPath(defStr:str):
	"""check if written def is valid path """
	if "." in defStr and "/" in defStr:
		return True
	return False

def hdaDefNameFromDefFile(defStr:str, edit=False)->str:
	if defIsPath(defStr):
		defStr = "FILE_" + Path(defStr).stem.split(".")[0]
	if edit:
		defStr = defStr + "_EDIT"
	return defStr + "_TextHDA"

def makePTGForParentParm(
		pt:hou.ParmTemplate,
)->hou.ParmTemplateGroup:
	"""add new ptg to represent single param -
	- PARENT_parmName is disabled display of exact value on parent class
	- OVERRIDE_parmName is checkbox to say whether this value is overridden
	- parmName is the actual parmName.

	how should we handle actually overriding parm names from parents in leaf?
	- at parent parm gen time, if parm with matching name is found in leaves,
		just defer to that

	"""
	ptg = hou.ParmTemplateGroup()
	leafParmName = pt.name()
	if "LEAF_" in leafParmName:
		leafParmName = leafParmName.replace("LEAF_", "")
	parentPt = pt.clone()
	parentPt.setName(f"PARENT_{leafParmName}")
	parentPt.setLabel("Parent value:")
	overridePt = hou.ToggleParmTemplate(
		f"OVERRIDE_{leafParmName}", "Override?")
	ptg.addParmTemplate(pt)
	ptg.addParmTemplate(overridePt)
	ptg.addParmTemplate(parentPt)
	return ptg

# def syncHDAParentParms(
# 		node:hou.OpNode,
# ):
# 	""""""
# 	hda = TextHDANode(node)
# 	for parentDef in hda.parentDefs():
# 		hdaDef = getHda

def getLeafDatasByParent(node:hou.OpNode)->dict[str, dict]:
	"""for each def in node's history,
	return that def's leaf state. stupid basic for now"""
	hda = TextHDANode(node)
	for i in hda.parentDefPaths():
		hdaDef = getHDADefFromPath(i)
		if hdaDef is None:
			continue
		leafData = getHDASectionDict(
			hdaDef, HDA_FULL_SECTION_NAME, default={}
		)
		yield i, leafData

def updateHDADefSections(
		hdaDef:hou.HDADefinition,
		wholeNodeState:dict,
		leafDelta:dict,
):
	setHDASectionDict(hdaDef, HDA_DELTA_SECTION_NAME,
	                         leafDelta)
	setHDASectionDict(hdaDef, HDA_FULL_SECTION_NAME,
	                         wholeNodeState)


def parmTemplateDialogScript(pt:hou.ParmTemplate)->str:
	"""return dialog script for given parm template"""
	ptg = hou.ParmTemplateGroup([pt])
	return makeSafeForJson(ptg.asDialogScript())

class TextHDAWorkContext:
	"""context to only set working state on outermost level
	need to keep """
	def __init__(self, hda:TextHDANode):
		self.path = hda.node.path()
		self.isOuter = False

	def node(self)->hou.Node:
		node = hou.node(self.path)
		assert node
		return node
	def __enter__(self):
		hda = TextHDANode(self.node())
		if not hda.isWorking():
			self.isOuter = True
			hda.setWorking(True)
		return self
	def __exit__(self, exc_type, exc_val, exc_tb):
		# if self.node() is None:
		# 	return
		hda = TextHDANode(self.node())
		if self.isOuter:
			hda.setWorking(False)

class TextHDANode:
	"""
	helper wrapper for TextHDA
	expose getting incoming state

	each node may have a defined hda archetype,
	and a temporary hda definition to allow editing
	"""

	def __init__(self, node:hou.OpNode):
		self.node = node

	"""use working state to prevent looping signals
	when internal processes change attributes"""
	@dbg
	def setWorking(self, state=True):
		print("setting working state to", state,self.node)
		self.node.setUserData("_working", str(int(state)))
		return
		if state:
			removeNodeInternalCallbacks(self.node)
		else:
			addNodeInternalCallbacks(self.node)
	def isWorking(self)->bool:
		return bool(int(self.node.userData("_working") or 0))
	def workCtx(self)->TextHDAWorkContext:
		ctx = TextHDAWorkContext(self)
		return ctx

	def lastHash(self)->int:
		return int(self.node.userData("_lastHash") or -1)
	def setLastHash(self, v:int):
		self.node.setUserData("_lastHash", str(v))
	def getHash(self)->int:
		"""run over this and all contained nodes and get their hash -
		use to avoid double-firing event signals
		"""
		# this is probably SUPER slow but for now it works
		data = self.node.asData(
			nodes_only=True,
			children=True,
			editables=True,
			inputs=True,
			position=True,
			flags=True,
			parms=True,
			default_parmvalues=False,
			evaluate_parmvalues=False,
			parms_as_brief=True,
			parmtemplates="spare_only",
			metadata=False,
			verbose=False
		)
		return hash(str(data))

	def defFileParm(self)->hou.Parm:
		return self.node.parm(ParmNames.defFile)
	def defFile(self)->Path|str:
		raw = self.defFileParm().evalAsString().strip()
		if not raw:
			return None

		return Path(raw) if defIsPath(raw) else raw

	def allDefsUsed(self)->list[str]:
		result = [self.defFile()] + self.parentDefPaths()

	def editingAllowedParm(self)->hou.Parm:
		return self.node.parm(ParmNames.allowEditing)
	def editingAllowed(self)->bool:
		return self.editingAllowedParm().eval()

	def isBaselineTextHDA(self)->bool:
		return self.editingAllowed() or self.parentStoredStates()

	def nParentSources(self)->int:
		parentFolder: hou.Parm = self.node.parm("parentfolder")
		nFolderParms = 3
		return len(parentFolder.multiParmInstances()) // nFolderParms

	def _nodeCachedParentState(self)->dict|None:
		return loads(self.node.userData("cachedParentState") or "{}")
	def _setNodeCachedParentState(self, data):
		self.node.setUserData("cachedParentState", dumps(data)
		                      )
	@dbg
	def getCachedParentState(self):
		#print("get cached parent state")
		if self._nodeCachedParentState() is None:
			storedStates = self.parentStoredStates()
			#print("stored states:", storedStates)
			self._setNodeCachedParentState(
				mergeNodeStates({}, storedStates)
			)
		return self._nodeCachedParentState()

	def hdaDef(self)->hou.HDADefinition:
		return self.node.type().definition()

	def nodeLeafDeltaParm(self)->hou.Parm:
		return self.node.parm(ParmNames.localEdits)
	def nodeLeafDeltaData(self)->dict:
		"""
		return local edited state stored in this node's params
		"""
		parmS = self.nodeLeafDeltaParm().evalAsString().lstrip().rstrip()
		if not parmS:
			return {}
		try:
			data = loads(parmS)
		except Exception as e:
			print("could not load local node edits, "
			      "check that the data is valid json")
			print(e)
			return {}
		return data
	def setNodeLeafDeltaData(self, data:dict):
		"""
		store local edited state in this node's params
		"""
		parmS = dumps(data)
		self.nodeLeafDeltaParm().set(parmS)

	def nodeLeafStoredState(self)->dict:
		"""
		return local edited state stored in this node's params
		"""
		parmS = self.node.parm(ParmNames.localEdits).evalAsString().lstrip().rstrip()
		if not parmS:
			return {}
		try:
			data = loads(parmS)
		except Exception as e:
			print("could not load local node edits, "
			      "check that the data is valid json")
			print(e)
			return {}
		return data

	def nodeLeafPath(self)->Path:
		s = self.node.parm(ParmNames.defFile).eval()
		if not s:
			return None
		return Path(s)

	def leafHDAParmFolderTemplate(self)->hou.FolderParmTemplate:
		return self.node.parmTemplateGroup().findFolder(ParmNames.leafHDAParmFolderLABEL)

	def leafHDAParmTemplates(self)->list[hou.ParmTemplate]:
		return self.leafHDAParmFolderTemplate().parmTemplates()
	def leafPTGData(self)->str:
		ptg = hou.ParmTemplateGroup(self.leafHDAParmTemplates())
		return ptg.asDialogScript(
			full_info=True
		)

	def parentHDAParmFolderTemplate(self)->hou.FolderParmTemplate:
		return self.node.parmTemplateGroup().findFolder(ParmNames.parentHDAParmFolderLABEL)

	def parentHDAParmTemplates(self)->list[hou.ParmTemplate]:
		return self.parentHDAParmFolderTemplate().parmTemplates()

	def leafParmTuples(self)->list[hou.ParmTuple]:
		return self.node.parmTuplesInFolder("Leaf HDA parms")

	def inheritedParmTupleMap(self)->dict[dict[str, hou.ParmTuple]]:
		"""map of inherited parm tuples by source def name"""
		result = defaultdict(dict)
		for pt in self.node.parmTuples():
			containing = pt.containingFolders()
			if not containing:
				continue
			if containing[0] == ParmNames.leafHDAParmFolderLABEL:
				if not self.defFile():
					continue
				result[str(self.defFile())][pt.name()] = pt
				continue
			if len(containing) != 2:
				continue
			# holds folder for each parent def
			parentDefName = containing[-1]
			result[parentDefName][pt.name()] = pt
		return result

	def parentParmTuples(self)->list[hou.ParmTuple]:
		"""unstructured, ungrouped etc"""
		result = []
		for k, v in self.inheritedParmTupleMap().items():
			if k == str(self.defFile()):
				continue
			result.extend(v.values())
		return result

	def inheritedParmTemplateMap(self)->dict[dict[str, hou.ParmTemplate]]:
		"""map of inherited parm tuples by source def name
		we save each parm separately so that we can split up for override

		{ def name : { parm name : parm template } }
		"""
		result = defaultdict(dict)
		for k, v in self.inheritedParmTupleMap().items():
			for ptName, pt in v.items():
				result[k][ptName] = pt.parmTemplate()
		return result

	def inheritedParmDialogScriptMap(self)->dict[dict[str, str]]:
		result = defaultdict(dict)
		for k, v in self.inheritedParmTupleMap().items():
			for ptName, pt in v.items():
				ptemplate : hou.ParmTemplate = pt.parmTemplate()
				result[k][ptName] = parmTemplateDialogScript(ptemplate)
		return result

	def statusParm(self)->hou.Parm:
		return self.node.parm("debuglabel")
	def setStatusMsg(self, msg:str):
		self.statusParm().set(msg)

	""" try to get some handle on nightmare of this system's state"""
	def syncEditingAllowed(self):
		"""check if editing checkbox should be enabled:
		- if the current node has a def
		"""

		if self.defFile():
			self.editingAllowedParm().disable(False) # not disabled
		else:
			self.editingAllowedParm().disable(True) # yes disabled

	# mutating functions
	def setLeafEditsLocked(self, state=True):
		"""mute any leaf-level edits, reset to incoming,
		but keep leaf delta parm data locked"""

	def getCustomHDADef(self)->hou.HDADefinition:
		"""create def if node doesn't already have one, then return it

		"""
		if ((not self.hdaDef()) or self.hdaDef() == getBaseTextHDADef()):
			newDef, newNode = createLocalTextHDADefinition(
				self.node, newId=self.node.name() + "_EDIT_TextHDA")
			self.node = newNode
		return self.hdaDef()

	@dbg
	def fullReset(self, resetParms=False):
		"""reset node to base textHDA definition
		ok FUN FACT, destroy() will absolutely delete the file of an HDA's
		definition

		"""
		#print("fullReset:", self.node, resetParms)
		leafDef = self.hdaDef()
		masterDef = getBaseTextHDADef()
		assert masterDef
		if leafDef == masterDef:
			print("hda def is already master")
		else:
			newNode = self.node.changeNodeType(
				masterDef.nodeTypeName(),
				keep_name=True,
				keep_parms=not resetParms,
				keep_network_contents=False
			)
			self.node = newNode
			# remove old def from scene
			deleteHDADefIfUnused(leafDef)
			self.editingAllowedParm().disable(False)
		#self.defFileParm().set("")
		#self.editingAllowedParm().set(False)

	nFolderParms = 3

	def nParentEntries(self)->int:
		parentFolder: hou.Parm = self.node.parm("parentfolder")
		return len(parentFolder.multiParmInstances()) // self.nFolderParms

	def _parentParms(self, parmName:str)->list[hou.Parm]:
		"""
		multi-instance folders are very annoying to work with
		multiparminstances returns ALL flat parametres under this folder
		intense pain
		"""
		results = []
		for i in range(self.nParentEntries()):
			results.append(self.node.parm(
				parmName + str(i + 1)))
		return results

	def parentDefPaths(self)->list[Path|str]:
		strs = [i.evalAsString() for i in self._parentParms(
			ParmNames.parentDef) if i.evalAsString().strip()]
		return [Path(i) if defIsPath(i) else i for i in strs]

	def parentDefs(self)->list[hou.HDADefinition]:
		result = []
		for i in self.parentDefPaths():
			hdaDef = getHDADefFromPath(i)
			if hdaDef is not None:
				result.append(hdaDef)
		return result

	def parentBaseDatas(self)->list[ParentBaseData]:
		filePaths = self._parentParms(ParmNames.parentFile)
		parentDeltas = self._parentParms(ParmNames.parentNodeDelta)
		parentLocalOverrides = self._parentParms(ParmNames.parentNodeLocalOverride)
		return [
			ParentBaseData(file=filePaths[i].evalAsString(),
			               text=parentDeltas[i].evalAsString(),
			               localOverride=parentLocalOverrides[i].evalAsString())
			for i in range(self.nParentEntries())
		]


	def parentStoredStates(self)->list[dict]:
		"""load parent states from parm text boxes -
		DOES NOT RELOAD source files"""
		return [
			loads(i.evalAsString()) or {}
			for i in self._parentParms(ParmNames.parentNodeDelta)
		]

	def filteredParentStoredStates(self)->list[dict]:
		return [
			loads(i.evalAsString())
			for i in self._parentParms(ParmNames.parentNodeDelta)
			if loads(i.evalAsString())
		]

	def hasIncomingStates(self)->bool:
		"""check if node has any incoming data from parents or leaf"""
		for i in self.parentStoredStates() + [self.nodeLeafStoredState()]:
			if any(i.values()):
				return True
		return False

	@dbg
	def reloadParentStates(self):
		parentStates = []
		parentParms = self._parentParms(ParmNames.parentNodeDelta)
		log("PARENT DEFS:", self.parentDefPaths())
		sceneHdaMap = getSceneHDADefNodes()
		for i, path in enumerate(self.parentDefPaths()):
			if not path:
				continue
			if isinstance(path, Path):
				if not path.is_file():
					# nodes can't set warnings on other badges
					# node.addWarning("Could not find parent file: " + str(i))
					continue
				pathCachedFileMap = self.node.hdaModule().pathCachedFileMap
				if path not in pathCachedFileMap:
					pathCachedFileMap[path] = CachedFile(str(path))

				baseData = pathCachedFileMap[path].readJson()

			else:  # def is string, look up in scene
				hdaDef = sceneHdaMap.get(path)
				log("hdaDef for", path, "is", hdaDef)
				if not hdaDef:
					continue
				hdaDef = hdaDef[0].type().definition()
				baseData = getHDASectionDict(
					hdaDef,
					#HDA_DELTA_SECTION_NAME,
					HDA_FULL_SECTION_NAME,
					{}
				)
				log("loaded base data for", path, "is:")
				pprint.pprint(baseData)
				if not baseData:
					continue

			parentStates.append(baseData)
			parentParms[i].set(dumps(baseData))

		incomingState = mergeNodeStates(
			{}, parentStates
		)
		self._setNodeCachedParentState(incomingState)
		return incomingState

	@dbg
	def syncOnlyNodeParmTemplates(self):
		incomingState = self.reloadParentStates()


	@dbg
	def syncNodeState(self, paramValuesOnly=False):
		"""update node from incoming and leaf data"""
		incomingState = self.reloadParentStates()
		if "LEAF" in incomingState["parmTemplates"]:
			# mark any incoming top-level params as parent
			if not "PARENT" in incomingState["parmTemplates"]:
				incomingState["parmTemplates"]["PARENT"] = {}
			incomingState["parmTemplates"]["PARENT"].update(
				incomingState["parmTemplates"]["LEAF"])
			incomingState["parmTemplates"].pop("LEAF")
		leafState = self.nodeLeafStoredState()
		combinedState = mergeNodeStates(incomingState, [leafState])
		setNodeToState(self.node, combinedState)

	@dbg
	def gatherSyncNodeState(self):
		"""fully re-gather and sync node"""

		incomingState = self.reloadParentStates()
		wholeNodeState = getFullNodeState(self.node)
		leafState = diffNodeState(incomingState, wholeNodeState)
		self.nodeLeafDeltaParm().set(dumps(leafState))



