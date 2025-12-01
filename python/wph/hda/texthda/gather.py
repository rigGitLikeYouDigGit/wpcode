from __future__ import annotations
import types, typing as T
import pprint
from pathlib import Path

from orjson import loads

from wplib import log

import copy, json, importlib
from typing import NamedTuple

from deepdiff import DeepDiff, Delta

import hou

from . import types
importlib.reload(types)

from .types import NodeHeader, ParmNames, CachedFile, dumps, loads


"""extract node deltas then params - 
for each node ref, save name and uid to allow both means

save the node's version mode and number as parametres, so if a node changes to an explicit version, that's a param change

save header,
meta,
params
"""

NETWORK_BOX_S = "NETWORK_BOX"
TOP_SEP_CHAR = "@@"
VERSION_SEP_CHAR = "@"


def truncate2Places(f:float):
	return int(f * 100) / 100.0

def getNodeHeader(node:hou.Node, rootNode:hou.Node):
	path = rootNode.relativePathTo(node)
	if isinstance(node, hou.NetworkBox):
		return NodeHeader(path, "NETWORK_BOX", "", "", "", 0)
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
	#defaultType = hou.NodeType(hou.sopNodeTypeCategory(), nodeTypeName )
	defaultType = hou.nodeType(hou.sopNodeTypeCategory(), nodeTypeName )
	exactVersionMatters = int(defaultType.nameComponents() != typeInfo.nameComponents())

	# save everything EXCEPT the version at node level
	return NodeHeader(path, scopeType, nodeTypeNS, nodeTypeName, exactVersion, exactVersionMatters,
	                  truncate2Places(node.position()[0]),
			truncate2Places(node.position()[1]))

def createNodeFromHeader(rootNode:hou.Node, header:NodeHeader):
	"""create a new node from the given header: path and node type.

	Looks like we need to know the version (and if it matters) at node creation,
	so pack that on the end of this header?"""
	nodePath, scopeType, nodeTypeNS, nodeTypeName, exactVersion, exactVersionMatters, x, y = header
	nodeName = nodePath.split("/")[-1]
	if "/" in nodePath:
		parentNodePath = "/".join(nodePath.split("/")[:-1])
	else:
		parentNodePath = "."
	parentNode : hou.Node = rootNode.node(parentNodePath)

	if exactVersionMatters:
		exactVersion = "::".join((nodeTypeNS, nodeTypeName, exactVersion))
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

# def getNodesDiff(baseData:set[NodeHeader], newData:set[NodeHeader])->dict[str, list[NodeHeader]]:
# 	"""return a dict of {"add" : [], "del" : [] }
# 	"""
# 	result = {"add" : [], "del" : []}
# 	for i in baseData:
# 		if not i in newData:
# 			result["del"].append(i)
# 	for i in newData:
# 		if not i in baseData:
# 			result["add"].append(i)
# 	return result


def getChildrenConnectionData(parentNode:hou.Node)->list[list[str]]:
	"""very simple, of form
	[ 0-myNodeOutputName-myNode , 0-myNodeInputName-myOtherNode ]"""
	connections = []
	for node in parentNode.children():
		node : hou.Node
		connectors : tuple[tuple[hou.NodeConnection]] = node.inputConnectors()
		for i, connector in enumerate(connectors):
			if not connector: # input not driven
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
		parentNode:hou.Node, connections:list[list[str]],
		useNames=True
):
	"""very simple, of form
	[ 0-myNodeOutputName-myNode , 0-myNodeInputName-myOtherNode ]

	TODO: add error checks throughout this to fall back if using names fails
	"""
	subnetInputs = parentNode.indirectInputs()
	for c in connections:
		dstIndex, dstInName, dstName = c[1].split("-")
		dstNode : hou.Node = parentNode.node(dstName)
		if len(c[0].split("-")) == 1: # subnet input
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
		baseData:list[list[str]],
		newData:set[NodeHeader]
)->dict[str, list[list[str]]]:
	"""return a dict of {"add" : [], "del" : [] }
	"""
	result = {"add" : [], "del" : []}
	for i in baseData:
		if not i in newData:
			result["del"].append(i)
	for i in newData:
		if not i in baseData:
			result["add"].append(i)
	return result


def getNodeParamText(node:hou.OpNode)->dict[str, T.Any]:
	"""get verbose so we can iterate over dictionary easier when diffing
	but no way to
	"""
	result = {}
	for i in node.parms():
		if i.isAtDefault():
			continue
		result[i.name()] = i.asData()
	return result

def setNodeParamsFromText(node:hou.OpNode, data):
	for parmName, v in data.items():
		node.parm(parmName).setFromData(v)


"""very quick and dirty deep diff and update system"""

def deepUpdatePath(baseData:dict|list, path:list, value):
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
def deepUpdate(baseData:dict|list, patch:list[tuple[list[str], T.Any]]):
	for k, v in patch:
		deepUpdatePath(baseData, list(k), v)


def deepDiffParams(
		baseData,
		newData
)->Delta:
	diff = DeepDiff(baseData, newData)
	return Delta(diff)

def applyDeepPatchParams(
		baseData,
		patch:Delta
):
	return baseData + patch

def diffParamsText(
		baseData:dict,
		newData:dict,
		patchData:dict=None
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


def getParmDialogScripts(node:hou.Node)->dict[str]:
	"""trying to save parm definitions in dialog script to ensure we
	keep everything -
	for some reason you can't get that for individual templates.

	So for each parm, make a new template group containing only it,
	and get that as dialog
	"""
	result = {}
	for p in node.spareParms():
		p : hou.Parm
		template = p.parmTemplate()
		ptg = hou.ParmTemplateGroup([template])
		result[p.name()] = ptg.asDialogScript()
	return result

def getTextHDAParmDialogScripts(node:hou.Node):
	"""special-case textHDA root nodes - maybe this isn't
	necessary, but I don't want the system accidentally
	erasing itself
	"""
	hda = TextHDANode(node)
	# leafPtg = hou.ParmTemplateGroup(hda.leafHDAParmTemplates())
	# parentPtg = hou.ParmTemplateGroup(hda.parentHDAParmTemplates())
	result = {
		"parent" : {i.name() : hou.ParmTemplateGroup(i).asDialogScript() for i in hda.parentHDAParmTemplates()},
		"leaf" : {i.name() : hou.ParmTemplateGroup(i).asDialogScript() for i in hda.leafHDAParmTemplates()},
	}
	return result

def setTextHDAParmDialogScripts(node:hou.Node, data:dict):
	"""remove existing hda parms and reset from given data"""
	hda = TextHDANode(node)
	ptg : hou.ParmTemplateGroup = node.parmTemplateGroup()

	parmNames = {"parent" : ParmNames.parentHDAParmFolder,
	             "leaf" : ParmNames.leafHDAParmFolder}
	for cat in ("parent", "leaf"):
		for k, v in data.get(cat, {}).items():
			parm = node.parm(k)
			if parm is not None: # exists on node
				try:
					ptg.remove(parm.parmTemplate())
				except hou.OperationFailed:
					pass

			parmPTG = hou.ParmTemplateGroup()
			parmPTG.setToDialogScript(v)
			folderPT : hou.FolderParmTemplate = ptg.findFolder(parmNames[cat])







def setParmDialogScripts(node:hou.Node, ptgData:dict):
	masterPtg : hou.ParmTemplateGroup = node.parmTemplateGroup()
	for parmName, data in ptgData.items():
		ptg = hou.ParmTemplateGroup()
		ptg.setToDialogScript(data)
		masterPtg.append(ptg.parmTemplates()[0])
	node.setParmTemplateGroup(masterPtg)


def iterNodesToTrack(topNode:hou.Node)->list[hou.Node]:
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


def getFullNodeState(
		node:hou.Node
)->dict:
	"""get full snapshot of node -
	prune at included text hda nodes
	"""
	print("getFullNodeState")
	nodes = iterNodesToTrack(node)
	print("nodes", nodes)
	# get params to add to the top node
	baseParmTemplateGroup : hou.ParmTemplateGroup = node.parmTemplateGroup()
	print("base ptg", baseParmTemplateGroup)
	#
	# for i, pt in tuple(
	# 		enumerate(baseParmTemplateGroup.parmTemplates())):
	# 	pt : hou.ParmTemplate
	# 	# remove any parms the hda will always have:
	# 	if any(s in pt.name() for s in paramsToIgnore):
	# 		try:
	# 			baseParmTemplateGroup.remove(pt)
	# 		except: # may try to remove params after containing folder already removed
	# 			pass
	#print("removed ignored parms")

	# parmTemplatesToAdd = {
	# 	"." : baseParmTemplateGroup.asDialogScript(full_info=True)
	# }
	#
	# parmTemplatesToAdd = {
	# 	"." :
	# }
	#print("pts to add:", parmTemplatesToAdd)
	parmTemplatesToAdd = {}
	for i in ([node] + nodes):
		if isTextHDANode(i):
			parmDialogData = getTextHDAParmDialogScripts(i)
		else:
			parmDialogData = getParmDialogScripts(i)
		if not parmDialogData:
			continue
		parmTemplatesToAdd[node.relativePathTo(i)] = parmDialogData

	print("pts to add:")
	pprint.pprint(parmTemplatesToAdd)

	parmVals = {}
	for i in nodes:
		parmValData = getNodeParamText(i)
		if not parmValData:
			continue
		parmVals[node.relativePathTo(i)] = parmValData

	# node paths are exclusive, so let those override
	result = {
		"nodes" : {node.relativePathTo(i) : getNodeHeader(i, node) for i in nodes},
		"parmTemplates" : parmTemplatesToAdd,
		#"connections" : [getChildrenConnectionData(i) for i in [node] + nodes],
		"connections" : getChildrenConnectionData(node),
		#"parmVals" : {node.relativePathTo(i) : getNodeParamText(i) for i in nodes}
		"parmVals" : parmVals
	}

	return result

"""we don't need sophisticated diffs for nodes, parm templates etc - 
overriding and adding should be fine, since dangling nodes
in the network LIKELY won't matter?

make that an option

"""

def mergeNodeStates(
		baseData:dict,
		states:list[dict]
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
			baseData["parmVals"][nodePath].update(parmData )

	return baseData


def diffNodeState(
		baseData:dict,
		wholeNodeState:dict
)->dict:
	"""get final dict to put in leaf parm
	works only as an overkay/override
	"""
	result = {
		"nodes" : {},
		"parmTemplates" : {},
		"connections" : [],
		"parmVals" : {}
	}
	# ensure baseData has all the needed keys set up
	for k, v in result.items():
		baseData.setdefault(k, v)
	print("diffNodeState:")
	print("base:")
	pprint.pprint(baseData)
	print("whole state:")
	pprint.pprint(wholeNodeState)
	for nodePath, nodeData in wholeNodeState.get("nodes", {}).items():
		"""override node creation if leaf node is not found,
		or if it has a different type/version compared to base"""
		if str(nodeData) != str(baseData["nodes"].get(nodePath, "")):
			result["nodes"][nodePath] = nodeData

	for nodePath, parmData in wholeNodeState.get("parmTemplates", {}).items():
		if not nodePath in baseData["parmTemplates"]:
			result["parmTemplates"][nodePath] = parmData
			continue
		for parmName, parmScript in parmData.items():
			if str(parmScript) != str(baseData["parmTemplates"][nodePath].get(parmName, "")):
				result["parmTemplates"][nodePath][parmName] = parmScript

	comp = set(map(tuple, baseData.get("connections", [])))
	for i in wholeNodeState.get("connections", []):
		if not tuple(i) in comp:
			result["connections"].append(i)

	for nodePath, parmData in wholeNodeState.get("parmVals", {}).items():
		if not nodePath in baseData["parmVals"]:
			result["parmVals"][nodePath] = parmData
			continue
		for parmName, parmVals in parmData.items():
			if not parmName in baseData["parmVals"][nodePath].get(parmName, {}):
				result["parmVals"][nodePath][parmName] = parmVals
	return result


def isTextHDANode(node:hou.Node):
	return node.type().nameComponents()[2].lower() in ("texthda", "textdeltahda")

def setNodeToState(
		node:hou.Node,
		state:dict
):
	"""conform node exactly to the given state - this WILL
	delete all non-tracked nodes and connections
	TODO: test if this is faster than going through more carefully item by item
	"""
	node.deleteItems(node.allSubChildren())
	for nodePath, nodeData in state["nodes"].items():
		createNodeFromHeader(node, nodeData)
	for nodePath, ptgData in state["parmTemplates"].items():
		setNode = node.node(nodePath)
		if isTextHDANode(setNode):
			setTextHDAParmDialogScripts(setNode, ptgData)
		else:
			setParmDialogScripts(node.node(nodePath), ptgData)
	setChildrenConnectionData(node, state["connections"])
	for nodePath, parmVals in state["parmVals"].items():
		setNodeParamsFromText(node.node(nodePath), parmVals )


class TextHDAWorkContext:
	"""context to only set working state on outermost level"""
	def __init__(self, hda:TextHDANode):
		self.hda = hda
		self.isOuter = False
	def __enter__(self):
		if not self.hda.isWorking():
			self.isOuter = True
			self.hda.setWorking(True)
	def __exit__(self, exc_type, exc_val, exc_tb):
		if self.isOuter:
			self.hda.setWorking(False)

class TextHDANode:
	"""
	helper wrapper for TextHDA
	expose getting incoming state
	"""

	def __init__(self, node:hou.Node):
		self.node = node

	"""use working state to prevent looping signals
	when internal processes change attributes"""
	def setWorking(self, state=True):
		self.node.setUserData("_working", str(int(state)))
	def isWorking(self)->bool:
		return bool(int(self.node.userData("_working") or 0))

	def workCtx(self)->TextHDAWorkContext:
		return TextHDAWorkContext(self)

	def editingAllowed(self)->bool:
		return self.node.parm(ParmNames.allowEditing).eval()

	def nParentSources(self)->int:
		parentFolder: hou.Parm = self.node.parm("parentfolder")
		nFolderParms = 3
		return len(parentFolder.multiParmInstances()) // nFolderParms

	def _nodeCachedParentState(self)->dict|None:
		return loads(self.node.userData("cachedParentState") or "{}")
	def _setNodeCachedParentState(self, data):
		self.node.setUserData("cachedParentState", dumps(data)
		                      )

	def getCachedParentState(self):
		print("get cached parent state")
		if self._nodeCachedParentState() is None:
			storedStates = self.parentStoredStates()
			print("stored states:", storedStates)
			self._setNodeCachedParentState(
				mergeNodeStates({}, storedStates)
			)
		return self._nodeCachedParentState()


	def _parentParms(self, parmName:str)->list[hou.Parm]:
		"""
		multi-instance folders are very annoying to work with
		multiparminstances returns ALL flat parametres under this folder,

		intense pain
		find a decent way to wrap this
		"""
		parentFolder: hou.Parm = self.node.parm("parentfolder")
		nFolderParms = 3
		results = []
		for i in range(len(parentFolder.multiParmInstances()) // nFolderParms):
			results.append(self.node.parm(
				parmName + str(i + 1)))
		return results

	def parentPaths(self)->list[Path]:
		return [Path(i.eval()) for i in self._parentParms(ParmNames.parentFile)]

	def parentStoredStates(self)->list[dict]:
		"""load parent states from parm text boxes -
		DOES NOT RELOAD source files"""
		return [
			loads(i.evalAsString()) or {}
			for i in self._parentParms(ParmNames.parentNodeDelta)
		]

	def reloadParentStates(self):
		parentStates = []
		parentParms = self._parentParms(ParmNames.parentNodeDelta)
		for i, path in enumerate(self.parentPaths()):
			if not path.is_file():
				# nodes can't set warnings on other badges
				# node.addWarning("Could not find parent file: " + str(i))
				continue
			pathCachedFileMap = self.node.hdaModule().pathCachedFileMap
			if path not in pathCachedFileMap:
				pathCachedFileMap[path] = CachedFile(str(path))

			baseData = pathCachedFileMap[path].readJson()
			parentStates.append(baseData)
			parentParms[i].set(dumps(baseData))

		incomingState = mergeNodeStates(
			{}, parentStates
		)
		self._setNodeCachedParentState(incomingState)
		return incomingState

	def nodeLeafDeltaParm(self)->hou.Parm:
		return self.node.parm(ParmNames.localEdits)

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

	def leafHDAParmFolderTemplate(self)->hou.FolderParmTemplate:
		return self.node.parmTemplateGroup().findFolder(ParmNames.leafHDAParmFolderLABEL)

	def leafHDAParmTemplates(self)->list[hou.ParmTemplate]:
		return self.leafHDAParmFolderTemplate().parmTemplates()

	def parentHDAParmFolderTemplate(self)->hou.FolderParmTemplate:
		return self.node.parmTemplateGroup().findFolder(ParmNames.parentHDAParmFolderLABEL)

	def parentHDAParmTemplates(self)->list[hou.ParmTemplate]:
		return self.parentHDAParmFolderTemplate().parmTemplates()


