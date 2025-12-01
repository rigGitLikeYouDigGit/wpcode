

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
	node : PolyDuplicateEdge = None
	pass
class DeleteEdgePlug(Plug):
	node : PolyDuplicateEdge = None
	pass
class EndVertexOffsetPlug(Plug):
	node : PolyDuplicateEdge = None
	pass
class InsertWithEdgeFlowPlug(Plug):
	node : PolyDuplicateEdge = None
	pass
class OffsetPlug(Plug):
	node : PolyDuplicateEdge = None
	pass
class SmoothingAnglePlug(Plug):
	node : PolyDuplicateEdge = None
	pass
class SplitTypePlug(Plug):
	node : PolyDuplicateEdge = None
	pass
class StartVertexOffsetPlug(Plug):
	node : PolyDuplicateEdge = None
	pass
# endregion


# define node class
class PolyDuplicateEdge(PolyModifier):
	adjustEdgeFlow_ : AdjustEdgeFlowPlug = PlugDescriptor("adjustEdgeFlow")
	deleteEdge_ : DeleteEdgePlug = PlugDescriptor("deleteEdge")
	endVertexOffset_ : EndVertexOffsetPlug = PlugDescriptor("endVertexOffset")
	insertWithEdgeFlow_ : InsertWithEdgeFlowPlug = PlugDescriptor("insertWithEdgeFlow")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	smoothingAngle_ : SmoothingAnglePlug = PlugDescriptor("smoothingAngle")
	splitType_ : SplitTypePlug = PlugDescriptor("splitType")
	startVertexOffset_ : StartVertexOffsetPlug = PlugDescriptor("startVertexOffset")

	# node attributes

	typeName = "polyDuplicateEdge"
	apiTypeInt = 973
	apiTypeStr = "kPolyDuplicateEdge"
	typeIdInt = 1346655557
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["adjustEdgeFlow", "deleteEdge", "endVertexOffset", "insertWithEdgeFlow", "offset", "smoothingAngle", "splitType", "startVertexOffset"]
	nodeLeafPlugs = ["adjustEdgeFlow", "deleteEdge", "endVertexOffset", "insertWithEdgeFlow", "offset", "smoothingAngle", "splitType", "startVertexOffset"]
	pass

