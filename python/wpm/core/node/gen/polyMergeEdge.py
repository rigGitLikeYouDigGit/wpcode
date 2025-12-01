

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifier = retriever.getNodeCls("PolyModifier")
assert PolyModifier
if T.TYPE_CHECKING:
	from .. import PolyModifier

# add node doc



# region plug type defs
class FirstEdgePlug(Plug):
	node : PolyMergeEdge = None
	pass
class MergeModePlug(Plug):
	node : PolyMergeEdge = None
	pass
class MergeTexturePlug(Plug):
	node : PolyMergeEdge = None
	pass
class SecondEdgePlug(Plug):
	node : PolyMergeEdge = None
	pass
# endregion


# define node class
class PolyMergeEdge(PolyModifier):
	firstEdge_ : FirstEdgePlug = PlugDescriptor("firstEdge")
	mergeMode_ : MergeModePlug = PlugDescriptor("mergeMode")
	mergeTexture_ : MergeTexturePlug = PlugDescriptor("mergeTexture")
	secondEdge_ : SecondEdgePlug = PlugDescriptor("secondEdge")

	# node attributes

	typeName = "polyMergeEdge"
	apiTypeInt = 416
	apiTypeStr = "kPolyMergeEdge"
	typeIdInt = 1347241285
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["firstEdge", "mergeMode", "mergeTexture", "secondEdge"]
	nodeLeafPlugs = ["firstEdge", "mergeMode", "mergeTexture", "secondEdge"]
	pass

