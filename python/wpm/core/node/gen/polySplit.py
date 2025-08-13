

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
	node : PolySplit = None
	pass
class Clean2VertsPlug(Plug):
	node : PolySplit = None
	pass
class DescPlug(Plug):
	node : PolySplit = None
	pass
class DetachEdgesPlug(Plug):
	node : PolySplit = None
	pass
class EdgePlug(Plug):
	node : PolySplit = None
	pass
class InsertWithEdgeFlowPlug(Plug):
	node : PolySplit = None
	pass
class Maya2015Plug(Plug):
	node : PolySplit = None
	pass
class Maya70Plug(Plug):
	node : PolySplit = None
	pass
class ProjectedCurveTolerancePlug(Plug):
	node : PolySplit = None
	pass
class SmoothingAnglePlug(Plug):
	node : PolySplit = None
	pass
class BaryCoord1Plug(Plug):
	parent : BaryCoordPlug = PlugDescriptor("baryCoord")
	node : PolySplit = None
	pass
class BaryCoord2Plug(Plug):
	parent : BaryCoordPlug = PlugDescriptor("baryCoord")
	node : PolySplit = None
	pass
class BaryCoord3Plug(Plug):
	parent : BaryCoordPlug = PlugDescriptor("baryCoord")
	node : PolySplit = None
	pass
class BaryCoordPlug(Plug):
	parent : SplitPointPlug = PlugDescriptor("splitPoint")
	baryCoord1_ : BaryCoord1Plug = PlugDescriptor("baryCoord1")
	bc1_ : BaryCoord1Plug = PlugDescriptor("baryCoord1")
	baryCoord2_ : BaryCoord2Plug = PlugDescriptor("baryCoord2")
	bc2_ : BaryCoord2Plug = PlugDescriptor("baryCoord2")
	baryCoord3_ : BaryCoord3Plug = PlugDescriptor("baryCoord3")
	bc3_ : BaryCoord3Plug = PlugDescriptor("baryCoord3")
	node : PolySplit = None
	pass
class FacePlug(Plug):
	parent : SplitPointPlug = PlugDescriptor("splitPoint")
	node : PolySplit = None
	pass
class TrianglePlug(Plug):
	parent : SplitPointPlug = PlugDescriptor("splitPoint")
	node : PolySplit = None
	pass
class SplitPointPlug(Plug):
	parent : SplitPointsPlug = PlugDescriptor("splitPoints")
	baryCoord_ : BaryCoordPlug = PlugDescriptor("baryCoord")
	bc_ : BaryCoordPlug = PlugDescriptor("baryCoord")
	face_ : FacePlug = PlugDescriptor("face")
	f_ : FacePlug = PlugDescriptor("face")
	triangle_ : TrianglePlug = PlugDescriptor("triangle")
	t_ : TrianglePlug = PlugDescriptor("triangle")
	node : PolySplit = None
	pass
class SplitPointsPlug(Plug):
	splitPoint_ : SplitPointPlug = PlugDescriptor("splitPoint")
	sp_ : SplitPointPlug = PlugDescriptor("splitPoint")
	node : PolySplit = None
	pass
class SubdivisionPlug(Plug):
	node : PolySplit = None
	pass
class VtxxPlug(Plug):
	parent : VerticesPlug = PlugDescriptor("vertices")
	node : PolySplit = None
	pass
class VtxyPlug(Plug):
	parent : VerticesPlug = PlugDescriptor("vertices")
	node : PolySplit = None
	pass
class VtxzPlug(Plug):
	parent : VerticesPlug = PlugDescriptor("vertices")
	node : PolySplit = None
	pass
class VerticesPlug(Plug):
	vtxx_ : VtxxPlug = PlugDescriptor("vtxx")
	vx_ : VtxxPlug = PlugDescriptor("vtxx")
	vtxy_ : VtxyPlug = PlugDescriptor("vtxy")
	vy_ : VtxyPlug = PlugDescriptor("vtxy")
	vtxz_ : VtxzPlug = PlugDescriptor("vtxz")
	vz_ : VtxzPlug = PlugDescriptor("vtxz")
	node : PolySplit = None
	pass
# endregion


# define node class
class PolySplit(PolyModifier):
	adjustEdgeFlow_ : AdjustEdgeFlowPlug = PlugDescriptor("adjustEdgeFlow")
	clean2Verts_ : Clean2VertsPlug = PlugDescriptor("clean2Verts")
	desc_ : DescPlug = PlugDescriptor("desc")
	detachEdges_ : DetachEdgesPlug = PlugDescriptor("detachEdges")
	edge_ : EdgePlug = PlugDescriptor("edge")
	insertWithEdgeFlow_ : InsertWithEdgeFlowPlug = PlugDescriptor("insertWithEdgeFlow")
	maya2015_ : Maya2015Plug = PlugDescriptor("maya2015")
	maya70_ : Maya70Plug = PlugDescriptor("maya70")
	projectedCurveTolerance_ : ProjectedCurveTolerancePlug = PlugDescriptor("projectedCurveTolerance")
	smoothingAngle_ : SmoothingAnglePlug = PlugDescriptor("smoothingAngle")
	baryCoord1_ : BaryCoord1Plug = PlugDescriptor("baryCoord1")
	baryCoord2_ : BaryCoord2Plug = PlugDescriptor("baryCoord2")
	baryCoord3_ : BaryCoord3Plug = PlugDescriptor("baryCoord3")
	baryCoord_ : BaryCoordPlug = PlugDescriptor("baryCoord")
	face_ : FacePlug = PlugDescriptor("face")
	triangle_ : TrianglePlug = PlugDescriptor("triangle")
	splitPoint_ : SplitPointPlug = PlugDescriptor("splitPoint")
	splitPoints_ : SplitPointsPlug = PlugDescriptor("splitPoints")
	subdivision_ : SubdivisionPlug = PlugDescriptor("subdivision")
	vtxx_ : VtxxPlug = PlugDescriptor("vtxx")
	vtxy_ : VtxyPlug = PlugDescriptor("vtxy")
	vtxz_ : VtxzPlug = PlugDescriptor("vtxz")
	vertices_ : VerticesPlug = PlugDescriptor("vertices")

	# node attributes

	typeName = "polySplit"
	apiTypeInt = 431
	apiTypeStr = "kPolySplit"
	typeIdInt = 1347637324
	MFnCls = om.MFnDependencyNode
	pass

