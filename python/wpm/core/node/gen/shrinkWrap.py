

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
WeightGeometryFilter = retriever.getNodeCls("WeightGeometryFilter")
assert WeightGeometryFilter
if T.TYPE_CHECKING:
	from .. import WeightGeometryFilter

# add node doc



# region plug type defs
class AlongXPlug(Plug):
	node : ShrinkWrap = None
	pass
class AlongYPlug(Plug):
	node : ShrinkWrap = None
	pass
class AlongZPlug(Plug):
	node : ShrinkWrap = None
	pass
class AxisReferencePlug(Plug):
	node : ShrinkWrap = None
	pass
class BidirectionalPlug(Plug):
	node : ShrinkWrap = None
	pass
class BoundaryRulePlug(Plug):
	node : ShrinkWrap = None
	pass
class BoundingBoxCenterPlug(Plug):
	node : ShrinkWrap = None
	pass
class CachedInflatedTargetPlug(Plug):
	node : ShrinkWrap = None
	pass
class CachedSmoothTargetPlug(Plug):
	node : ShrinkWrap = None
	pass
class ClosestIfNoIntersectionPlug(Plug):
	node : ShrinkWrap = None
	pass
class ContinuityPlug(Plug):
	node : ShrinkWrap = None
	pass
class FalloffPlug(Plug):
	node : ShrinkWrap = None
	pass
class FalloffIterationsPlug(Plug):
	node : ShrinkWrap = None
	pass
class InnerGeomPlug(Plug):
	node : ShrinkWrap = None
	pass
class InnerGroupIdPlug(Plug):
	node : ShrinkWrap = None
	pass
class InputEnvelopePlug(Plug):
	node : ShrinkWrap = None
	pass
class KeepBorderPlug(Plug):
	node : ShrinkWrap = None
	pass
class KeepHardEdgePlug(Plug):
	node : ShrinkWrap = None
	pass
class KeepMapBordersPlug(Plug):
	node : ShrinkWrap = None
	pass
class OffsetPlug(Plug):
	node : ShrinkWrap = None
	pass
class ProjectionPlug(Plug):
	node : ShrinkWrap = None
	pass
class PropagateEdgeHardnessPlug(Plug):
	node : ShrinkWrap = None
	pass
class ReversePlug(Plug):
	node : ShrinkWrap = None
	pass
class ShapePreservationEnablePlug(Plug):
	node : ShrinkWrap = None
	pass
class ShapePreservationIterationsPlug(Plug):
	node : ShrinkWrap = None
	pass
class ShapePreservationMethodPlug(Plug):
	node : ShrinkWrap = None
	pass
class ShapePreservationReprojectionPlug(Plug):
	node : ShrinkWrap = None
	pass
class ShapePreservationStepsPlug(Plug):
	node : ShrinkWrap = None
	pass
class SmoothUVsPlug(Plug):
	node : ShrinkWrap = None
	pass
class TargetGeomPlug(Plug):
	node : ShrinkWrap = None
	pass
class TargetInflationPlug(Plug):
	node : ShrinkWrap = None
	pass
class TargetSmoothLevelPlug(Plug):
	node : ShrinkWrap = None
	pass
# endregion


# define node class
class ShrinkWrap(WeightGeometryFilter):
	alongX_ : AlongXPlug = PlugDescriptor("alongX")
	alongY_ : AlongYPlug = PlugDescriptor("alongY")
	alongZ_ : AlongZPlug = PlugDescriptor("alongZ")
	axisReference_ : AxisReferencePlug = PlugDescriptor("axisReference")
	bidirectional_ : BidirectionalPlug = PlugDescriptor("bidirectional")
	boundaryRule_ : BoundaryRulePlug = PlugDescriptor("boundaryRule")
	boundingBoxCenter_ : BoundingBoxCenterPlug = PlugDescriptor("boundingBoxCenter")
	cachedInflatedTarget_ : CachedInflatedTargetPlug = PlugDescriptor("cachedInflatedTarget")
	cachedSmoothTarget_ : CachedSmoothTargetPlug = PlugDescriptor("cachedSmoothTarget")
	closestIfNoIntersection_ : ClosestIfNoIntersectionPlug = PlugDescriptor("closestIfNoIntersection")
	continuity_ : ContinuityPlug = PlugDescriptor("continuity")
	falloff_ : FalloffPlug = PlugDescriptor("falloff")
	falloffIterations_ : FalloffIterationsPlug = PlugDescriptor("falloffIterations")
	innerGeom_ : InnerGeomPlug = PlugDescriptor("innerGeom")
	innerGroupId_ : InnerGroupIdPlug = PlugDescriptor("innerGroupId")
	inputEnvelope_ : InputEnvelopePlug = PlugDescriptor("inputEnvelope")
	keepBorder_ : KeepBorderPlug = PlugDescriptor("keepBorder")
	keepHardEdge_ : KeepHardEdgePlug = PlugDescriptor("keepHardEdge")
	keepMapBorders_ : KeepMapBordersPlug = PlugDescriptor("keepMapBorders")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	projection_ : ProjectionPlug = PlugDescriptor("projection")
	propagateEdgeHardness_ : PropagateEdgeHardnessPlug = PlugDescriptor("propagateEdgeHardness")
	reverse_ : ReversePlug = PlugDescriptor("reverse")
	shapePreservationEnable_ : ShapePreservationEnablePlug = PlugDescriptor("shapePreservationEnable")
	shapePreservationIterations_ : ShapePreservationIterationsPlug = PlugDescriptor("shapePreservationIterations")
	shapePreservationMethod_ : ShapePreservationMethodPlug = PlugDescriptor("shapePreservationMethod")
	shapePreservationReprojection_ : ShapePreservationReprojectionPlug = PlugDescriptor("shapePreservationReprojection")
	shapePreservationSteps_ : ShapePreservationStepsPlug = PlugDescriptor("shapePreservationSteps")
	smoothUVs_ : SmoothUVsPlug = PlugDescriptor("smoothUVs")
	targetGeom_ : TargetGeomPlug = PlugDescriptor("targetGeom")
	targetInflation_ : TargetInflationPlug = PlugDescriptor("targetInflation")
	targetSmoothLevel_ : TargetSmoothLevelPlug = PlugDescriptor("targetSmoothLevel")

	# node attributes

	typeName = "shrinkWrap"
	typeIdInt = 1398231632
	pass

