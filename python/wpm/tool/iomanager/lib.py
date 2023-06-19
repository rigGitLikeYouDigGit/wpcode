
from __future__ import annotations
import typing as T
import json

from pathlib import Path

from tree import Tree

from wpm import cmds, om, oma, WN, createWN
from wpm.lib import io

"""small libs to interface with tree data on nodes
for now we say that a group is only input or output, not both
"""

MODE_KEY = "mode"
MODE_IMPORT = "import"
MODE_EXPORT = "export"
IO_KEY = "ioPath"

# probably ok to have exportPath be a global key here,
# single maya nodes should only export to _out folders anyway
def isExportNode(node:WN)->bool:
	"""return True if node is a valid export node"""
	if IO_KEY in node.getAuxTree().keys():
		return node.getAuxTree()[MODE_KEY] == MODE_EXPORT
	return False

def isImportNode(node:WN)->bool:
	"""return True if node is a valid import node"""
	if IO_KEY in node.getAuxTree().keys():
		return node.getAuxTree()[MODE_KEY] == MODE_IMPORT
	return False

def nodeIoPath(node:WN)->Path:
	"""return path to io folder for this node"""
	return Path(node.getAuxTree()[IO_KEY])

def syncExportNode(node:WN):
	"""sync export node with current node state"""
	io.exportSimpleUsd(node.getAuxTree()[EXPORT_KEY], node)






