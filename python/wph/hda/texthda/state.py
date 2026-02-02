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
	but no way to
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


def setTextHDAParmDialogScripts(node: hou.Node, data: dict):
	"""remove existing hda parms and reset from given data
	we modify the current hda definition ptg used by the node
	"""
	# print("text parms")
	# print(data)
	hda = TextHDANode(node)
	ptg: hou.ParmTemplateGroup = hda.hdaDef().parmTemplateGroup()

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
		node: hou.Node
) -> dict:
	"""get full snapshot of node -
	prune at included text hda nodes
	"""
	hda = TextHDANode(node)
	nodes = iterNodesToTrack(node)
	print("nodes", nodes)
	# get params to add to the top node
	baseParmTemplateGroup: hou.ParmTemplateGroup = node.parmTemplateGroup()
	print("base ptg", baseParmTemplateGroup)

	# for i, pt in hda.leafHDAParmTemplates():
	# 	parmTemplatesToAdd = {
	# 		"." : baseParmTemplateGroup.asDialogScript(full_info=True)
	# 	}
	parmTemplatesToAdd = {}
	for i in ([node] + nodes):
		if isTextHDANode(i):
			parmDialogData = getTextHDAParmDialogScripts(i)
		else:
			parmDialogData = getParmDialogScripts(i)
		if not parmDialogData:
			continue
		parmTemplatesToAdd[node.relativePathTo(i)] = parmDialogData

	# print("pts to add:")
	# pprint.pprint(parmTemplatesToAdd)

	parmVals = {}
	for i in nodes:
		parmValData = getNodeParamText(i)
		if not parmValData:
			continue
		parmVals[node.relativePathTo(i)] = parmValData

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
	pprint.pprint(result)

	return result


"""we don't need sophisticated diffs for nodes, parm templates etc - 
overriding and adding should be fine, since dangling nodes
in the network LIKELY won't matter?

make that an option

"""


def mergeNodeStates(
		baseData: dict,
		states: list[dict]
):
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


