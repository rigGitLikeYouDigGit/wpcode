
from __future__ import annotations
import typing as T
import json

from pathlib import Path

from tree import Tree
from tree.lib.object import UidElement, TypeNamespace

from wpm import cmds, om, oma, WN, createWN
from wpm.lib import io

"""small libs to interface with tree data on nodes
for now we say that a group is only input or output, not both
"""

class IoMode(TypeNamespace):
	"""either import or export
	define some constants for whenever this concept appears in
	pipeline"""
	class _Base(TypeNamespace.base()):
		modeStr = "base"
		colour = (0, 0, 0)
		pass
	class Import(_Base):
		colour = (0.5, 0.5, 1) # blue input
		modeStr = "import"

	class Export(_Base):
		colour = (1, 0.7, 0.5) # orange output
		modeStr = "export"


MODE_KEY = "mode"
IO_KEY = "ioPath"

# probably ok to have exportPath be a global key here,
# single maya nodes should only export to _out folders anyway
def isExportNode(node:WN)->bool:
	"""return True if node is a valid export node"""
	if IO_KEY in node.getAuxTree().keys():
		return node.getAuxTree()[MODE_KEY] == IoMode.Export.modeStr
	return False

def isImportNode(node:WN)->bool:
	"""return True if node is a valid import node"""
	if IO_KEY in node.getAuxTree().keys():
		return node.getAuxTree()[MODE_KEY] == IoMode.Import.modeStr
	return False

def isIoNode(node:WN)->bool:
	"""return True if node is a valid io node"""
	return IO_KEY in node.getAuxTree().keys()

def nodeIoPath(node:WN)->Path:
	"""return path to io folder for this node"""
	return Path(node.getAuxTree()[IO_KEY])

def setIoNode(node:WN, path:Path, mode:str):
	"""set node to be io node"""
	node.getAuxTree()[IO_KEY] = str(path)
	node.getAuxTree()[MODE_KEY] = mode

def listIoNodes()->T.List[WN]:
	"""return all io nodes in scene"""
	return [n for n in cmds.ls("*", type="transform")
	        if isIoNode(WN(n))]

def listExportNodes()->T.List[WN]:
	"""return all export nodes in scene"""
	return [n for n in listIoNodes()
	        if isExportNode(n)]





