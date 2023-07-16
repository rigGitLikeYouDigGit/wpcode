
from __future__ import annotations
import typing as T
import json

from pathlib import Path, PurePath
from dataclasses import dataclass

from tree import Tree
from tree.lib.object import UidElement, TypeNamespace

from wp.constant import IoMode, getAssetRoot
from wp.validation import ValidationError, ValidationResult, Rule, RuleSet, ErrorReport
from wp.treefield import TreeField, TreeFieldParams

from wpm import cmds, om, oma, WN, createWN
from wpm.lib import io
from wpm.lib.usd import bridge

"""small libs to interface with tree data on nodes
for now we say that a group is only input or output, not both.

leaving function names with "import" and "export" rather than
"input" and "output" for now, since this lib is explicitly focused
on shipping data into and out of maya

"""

NODE_IO_KEY = "nodeIo"


# we need a proper way of prefixing / namespacing tool-specific data in nodes
MODE_KEY = "ioMode"
IO_KEY = "ioPath"
LIVE_KEY = "ioLive"

defaultPathMap = {
	IoMode.Input: "_in/geo",
	IoMode.Output: "_out/geo",
}

def defaultPathForMode(mode:IoMode.T())->Path:
	"""return default path for mode"""
	return Path(defaultPathMap[mode])


defaultMode = IoMode.Output

# tree field template for io data
ioTreeField = TreeField(NODE_IO_KEY)
ioTreeField.lookupCreate = True
ioTreeField("ioMode").value = defaultMode
ioTreeField("ioPath").value = defaultPathForMode(defaultMode)
ioTreeField("ioLive").value = False


def updateNodeAuxDataFromIoTree(node:WN, ioTree:Tree):
	"""update node aux data from io tree"""
	ioBranch : Tree = node.getAuxTree()(NODE_IO_KEY, create=True)
	ioBranch.update(ioTree)
	node.saveAuxTree()


@dataclass
class NodeIoData:
	"""dataclass to hold io data for a node"""
	path: Path
	mode: IoMode.T() = IoMode.Input
	live : bool = False

def nodeIoData(node:WN)->NodeIoData:
	"""return NodeIoData for node"""
	try:
		return NodeIoData(
			path=nodeIoPath(node),
			mode=IoMode[node.getAuxTree()[MODE_KEY]],
			live=node.getAuxTree().get(LIVE_KEY, False)
		)
	except LookupError:
		return None

def setNodeIoData(node:WN, data:NodeIoData):
	"""set NodeIoData for node"""
	setIoNode(node, data.mode, data.path, data.live)

# single maya nodes should only export to _out folders anyway

def nodeIoPath(node:WN)->Path:
	"""return path to io folder for this node"""
	return Path(node.getAuxTree()[IO_KEY])

def isIoNode(node:WN, mode:IoMode.T()=None)->bool:
	"""return True if node is a valid io node"""
	data = nodeIoData(node)
	if data is None:
		return False
	if mode is None:
		return True
	return mode == data.mode

def isExportNode(node:WN)->bool:
	"""return True if node is a valid export node"""
	return isIoNode(node, IoMode.Output)

def isImportNode(node:WN)->bool:
	"""return True if node is a valid import node"""
	return isIoNode(node, IoMode.Input)


def setIoNode(node:WN, mode:IoMode.T(), path:Path=None, live=False):
	"""set node to be io node"""
	path = path if path is not None else defaultPathForMode(mode)
	node.getAuxTree()(IO_KEY, create=True).setValue(str(path))
	node.getAuxTree()(MODE_KEY, create=True).setValue(mode.clsName())
	node.getAuxTree()(LIVE_KEY, create=True).setValue(live)
	node.saveAuxTree()

def listIoNodes()->T.List[WN]:
	"""return all io nodes in scene"""
	return [WN(n) for n in cmds.ls("*", type="transform")
	        if isIoNode(WN(n))]

def listExportNodes()->T.List[WN]:
	"""return all export nodes in scene"""
	return [n for n in listIoNodes()
	        if isExportNode(n)]

def listImportNodes()->T.List[WN]:
	"""return all export nodes in scene"""
	return [n for n in listIoNodes()
	        if isImportNode(n)]

def removeIoNode(node:WN):
	"""remove io data from node"""
	node.getAuxTree().remove(IO_KEY)
	node.getAuxTree().remove(MODE_KEY)
	node.saveAuxTree()

def createIoNode(mode:IoMode.T()=IoMode.Input, name=f"newIO_GRP")->WN:
	"""create new io node"""
	tf = createWN("transform", name)
	setIoNode(tf, mode, defaultPathForMode(mode))
	return tf

def getScenePath()->Path:
	"""return path to current maya scene"""
	if not cmds.file(q=True, sn=True):
		return None
	return Path(cmds.file(q=True, sn=True))

def runInput(node:WN, targetPath:Path=None):
	"""run input on node"""
	targetPath = targetPath if targetPath is not None else nodeIoPath(node)

def runOutput(node:WN, targetPath:Path=None):
	"""run output on node"""
	targetPath = Path(targetPath if targetPath is not None else nodeIoPath(node))
	assert cmds.file(q=True, sn=True), "Maya scene has no path on disk"
	# check that we output to a usda file
	if targetPath.suffix != ".usda":
		targetPath = targetPath.with_suffix(".usda")

	# make path relative to scene
	targetPath = getScenePath().parent / targetPath
	print("outputting to", targetPath)
	bridge.saveMayaNodeToUsd(node.MFn, stage=None, filePath=targetPath)
	print("output done")

# validation systems to check io paths
class PathIsRelativeRule(Rule):
	"""check that path is relative to asset root"""
	def checkInput(self, data: str) -> bool:
		if Path(data).is_absolute():
			raise ValidationError(f"Path {data} is not relative to asset root")
		return True

class MayaHasScenePathRule(Rule):
	"""Check that the current maya scene has a path on disk"""
	def checkInput(self, data: str) -> bool:
		if not cmds.file(q=True, sn=True):
			raise ValidationError(f"Maya scene has no path on disk")
		return True

ioPathValidationRuleSet = RuleSet(
	[PathIsRelativeRule(), MayaHasScenePathRule()]
)







