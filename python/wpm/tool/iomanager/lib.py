
from __future__ import annotations
import typing as T
import json

from tree import Tree

from wpm import cmds, om, oma

"""small libs to interface with tree data on nodes"""

AUX_ATTR_NAME = "wpAux"

def nodeTemplateAuxData(node)->Tree:
	return Tree("root")

def nodeAuxData(node)->Tree:
	"""return the aux data on the given node -
	replace once WNode is implemented
	"""
	if not cmds.attributeQuery(AUX_ATTR_NAME, node=node, exists=True):
		return None
	return node.get(AUX_ATTR_NAME, {})



