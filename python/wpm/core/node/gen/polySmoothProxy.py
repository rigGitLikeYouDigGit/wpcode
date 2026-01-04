

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifier = Catalogue.PolyModifier
else:
	from .. import retriever
	PolyModifier = retriever.getNodeCls("PolyModifier")
	assert PolyModifier

# add node doc



# region plug type defs
class BoundaryRulePlug(Plug):
	node : PolySmoothProxy = None
	pass
class CachedSmoothMeshPlug(Plug):
	node : PolySmoothProxy = None
	pass
class ContinuityPlug(Plug):
	node : PolySmoothProxy = None
	pass
class DegreePlug(Plug):
	node : PolySmoothProxy = None
	pass
class DivisionsPerEdgePlug(Plug):
	node : PolySmoothProxy = None
	pass
class ExponentialLevelPlug(Plug):
	node : PolySmoothProxy = None
	pass
class KeepBorderPlug(Plug):
	node : PolySmoothProxy = None
	pass
class KeepHardEdgePlug(Plug):
	node : PolySmoothProxy = None
	pass
class KeepMapBordersPlug(Plug):
	node : PolySmoothProxy = None
	pass
class LinearLevelPlug(Plug):
	node : PolySmoothProxy = None
	pass
class Maya2008AbovePlug(Plug):
	node : PolySmoothProxy = None
	pass
class Maya65AbovePlug(Plug):
	node : PolySmoothProxy = None
	pass
class MethodPlug(Plug):
	node : PolySmoothProxy = None
	pass
class MultiEdgeCreasePlug(Plug):
	node : PolySmoothProxy = None
	pass
class OsdCreaseMethodPlug(Plug):
	node : PolySmoothProxy = None
	pass
class OsdFvarBoundaryPlug(Plug):
	node : PolySmoothProxy = None
	pass
class OsdFvarPropagateCornersPlug(Plug):
	node : PolySmoothProxy = None
	pass
class OsdIndependentUVChannelsPlug(Plug):
	node : PolySmoothProxy = None
	pass
class OsdSmoothTrianglesPlug(Plug):
	node : PolySmoothProxy = None
	pass
class OsdVertBoundaryPlug(Plug):
	node : PolySmoothProxy = None
	pass
class PropagateEdgeHardnessPlug(Plug):
	node : PolySmoothProxy = None
	pass
class PushStrengthPlug(Plug):
	node : PolySmoothProxy = None
	pass
class RoundnessPlug(Plug):
	node : PolySmoothProxy = None
	pass
class SmoothUVsPlug(Plug):
	node : PolySmoothProxy = None
	pass
class SubdivisionTypePlug(Plug):
	node : PolySmoothProxy = None
	pass
class UseOsdBoundaryMethodsPlug(Plug):
	node : PolySmoothProxy = None
	pass
# endregion


# define node class
class PolySmoothProxy(PolyModifier):
	boundaryRule_ : BoundaryRulePlug = PlugDescriptor("boundaryRule")
	cachedSmoothMesh_ : CachedSmoothMeshPlug = PlugDescriptor("cachedSmoothMesh")
	continuity_ : ContinuityPlug = PlugDescriptor("continuity")
	degree_ : DegreePlug = PlugDescriptor("degree")
	divisionsPerEdge_ : DivisionsPerEdgePlug = PlugDescriptor("divisionsPerEdge")
	exponentialLevel_ : ExponentialLevelPlug = PlugDescriptor("exponentialLevel")
	keepBorder_ : KeepBorderPlug = PlugDescriptor("keepBorder")
	keepHardEdge_ : KeepHardEdgePlug = PlugDescriptor("keepHardEdge")
	keepMapBorders_ : KeepMapBordersPlug = PlugDescriptor("keepMapBorders")
	linearLevel_ : LinearLevelPlug = PlugDescriptor("linearLevel")
	maya2008Above_ : Maya2008AbovePlug = PlugDescriptor("maya2008Above")
	maya65Above_ : Maya65AbovePlug = PlugDescriptor("maya65Above")
	method_ : MethodPlug = PlugDescriptor("method")
	multiEdgeCrease_ : MultiEdgeCreasePlug = PlugDescriptor("multiEdgeCrease")
	osdCreaseMethod_ : OsdCreaseMethodPlug = PlugDescriptor("osdCreaseMethod")
	osdFvarBoundary_ : OsdFvarBoundaryPlug = PlugDescriptor("osdFvarBoundary")
	osdFvarPropagateCorners_ : OsdFvarPropagateCornersPlug = PlugDescriptor("osdFvarPropagateCorners")
	osdIndependentUVChannels_ : OsdIndependentUVChannelsPlug = PlugDescriptor("osdIndependentUVChannels")
	osdSmoothTriangles_ : OsdSmoothTrianglesPlug = PlugDescriptor("osdSmoothTriangles")
	osdVertBoundary_ : OsdVertBoundaryPlug = PlugDescriptor("osdVertBoundary")
	propagateEdgeHardness_ : PropagateEdgeHardnessPlug = PlugDescriptor("propagateEdgeHardness")
	pushStrength_ : PushStrengthPlug = PlugDescriptor("pushStrength")
	roundness_ : RoundnessPlug = PlugDescriptor("roundness")
	smoothUVs_ : SmoothUVsPlug = PlugDescriptor("smoothUVs")
	subdivisionType_ : SubdivisionTypePlug = PlugDescriptor("subdivisionType")
	useOsdBoundaryMethods_ : UseOsdBoundaryMethodsPlug = PlugDescriptor("useOsdBoundaryMethods")

	# node attributes

	typeName = "polySmoothProxy"
	apiTypeInt = 944
	apiTypeStr = "kPolySmoothProxy"
	typeIdInt = 1347636560
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["boundaryRule", "cachedSmoothMesh", "continuity", "degree", "divisionsPerEdge", "exponentialLevel", "keepBorder", "keepHardEdge", "keepMapBorders", "linearLevel", "maya2008Above", "maya65Above", "method", "multiEdgeCrease", "osdCreaseMethod", "osdFvarBoundary", "osdFvarPropagateCorners", "osdIndependentUVChannels", "osdSmoothTriangles", "osdVertBoundary", "propagateEdgeHardness", "pushStrength", "roundness", "smoothUVs", "subdivisionType", "useOsdBoundaryMethods"]
	nodeLeafPlugs = ["boundaryRule", "cachedSmoothMesh", "continuity", "degree", "divisionsPerEdge", "exponentialLevel", "keepBorder", "keepHardEdge", "keepMapBorders", "linearLevel", "maya2008Above", "maya65Above", "method", "multiEdgeCrease", "osdCreaseMethod", "osdFvarBoundary", "osdFvarPropagateCorners", "osdIndependentUVChannels", "osdSmoothTriangles", "osdVertBoundary", "propagateEdgeHardness", "pushStrength", "roundness", "smoothUVs", "subdivisionType", "useOsdBoundaryMethods"]
	pass

