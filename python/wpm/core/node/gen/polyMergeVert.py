

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifierWorld = retriever.getNodeCls("PolyModifierWorld")
assert PolyModifierWorld
if T.TYPE_CHECKING:
	from .. import PolyModifierWorld

# add node doc



# region plug type defs
class AlwaysMergeTwoVerticesPlug(Plug):
	node : PolyMergeVert = None
	pass
class DistancePlug(Plug):
	node : PolyMergeVert = None
	pass
class MergeToComponentsPlug(Plug):
	node : PolyMergeVert = None
	pass
class TexturePlug(Plug):
	node : PolyMergeVert = None
	pass
# endregion


# define node class
class PolyMergeVert(PolyModifierWorld):
	alwaysMergeTwoVertices_ : AlwaysMergeTwoVerticesPlug = PlugDescriptor("alwaysMergeTwoVertices")
	distance_ : DistancePlug = PlugDescriptor("distance")
	mergeToComponents_ : MergeToComponentsPlug = PlugDescriptor("mergeToComponents")
	texture_ : TexturePlug = PlugDescriptor("texture")

	# node attributes

	typeName = "polyMergeVert"
	apiTypeInt = 698
	apiTypeStr = "kPolyMergeVert"
	typeIdInt = 1347245637
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["alwaysMergeTwoVertices", "distance", "mergeToComponents", "texture"]
	nodeLeafPlugs = ["alwaysMergeTwoVertices", "distance", "mergeToComponents", "texture"]
	pass

