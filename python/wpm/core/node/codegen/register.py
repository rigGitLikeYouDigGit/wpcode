from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import orjson

from wpm import cmds, om

# TEMP
pluginDataPath = "C:/Users/ed/Documents/GitHub/wpcode/python/wpm/core/node/codegen/pluginNodeData.json"

def readNodeData() -> T.Dict[str, T.Dict]:
	"""read node data from disk"""
	with open(pluginDataPath, "rb") as f:
		data = orjson.loads(f.read())
	return data

def registerNodeType(nodeTypeName:str):
	"""test for a friendlier interface to the codegen system -
	whenever a plugin is registered, re-sync all files associated
	with node types in that plugin
	"""

	ncls = om.MNodeClass(nodeTypeName)
	assert ncls.typeId



