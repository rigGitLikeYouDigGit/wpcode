
from __future__ import annotations
import typing as T
from pathlib import Path

from wpm import cmds, om, oma, WN, createWN
"""lib to manage consistent import and export functions"""

def importSimpleUsd(path:Path, parent:WN=None)->WN:
	"""import usd file, return top level node.
	if parent is given, import under that node.

	No USD filtering done at all, import everything
	"""

	# if parent:
	# 	cmds.file(path, i=True, parentReference=parent)
	# else:
	# 	cmds.file(path, i=True)
	#
	# return WN(cmds.ls(sl=True)[0])

def exportSimpleUsd(path:Path, node:WN):
	"""export node and all children to usd file"""
	cmds.select(node)
	cmds.file(path, exportSelected=True, type="USD Export")


