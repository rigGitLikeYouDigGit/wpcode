

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
class AdjustEdgeFlowPlug(Plug):
	node : PolyEditEdgeFlow = None
	pass
class EdgeFlowPlug(Plug):
	node : PolyEditEdgeFlow = None
	pass
# endregion


# define node class
class PolyEditEdgeFlow(PolyModifier):
	adjustEdgeFlow_ : AdjustEdgeFlowPlug = PlugDescriptor("adjustEdgeFlow")
	edgeFlow_ : EdgeFlowPlug = PlugDescriptor("edgeFlow")

	# node attributes

	typeName = "polyEditEdgeFlow"
	apiTypeInt = 1091
	apiTypeStr = "kPolyEditEdgeFlow"
	typeIdInt = 1347634502
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["adjustEdgeFlow", "edgeFlow"]
	nodeLeafPlugs = ["adjustEdgeFlow", "edgeFlow"]
	pass

