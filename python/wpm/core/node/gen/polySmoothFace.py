

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
	node : PolySmoothFace = None
	pass
class ContinuityPlug(Plug):
	node : PolySmoothFace = None
	pass
class DegreePlug(Plug):
	node : PolySmoothFace = None
	pass
class DivisionsPlug(Plug):
	node : PolySmoothFace = None
	pass
class DivisionsPerEdgePlug(Plug):
	node : PolySmoothFace = None
	pass
class KeepBorderPlug(Plug):
	node : PolySmoothFace = None
	pass
class KeepHardEdgePlug(Plug):
	node : PolySmoothFace = None
	pass
class KeepMapBordersPlug(Plug):
	node : PolySmoothFace = None
	pass
class KeepSelectionBorderPlug(Plug):
	node : PolySmoothFace = None
	pass
class KeepTessellationPlug(Plug):
	node : PolySmoothFace = None
	pass
class Maya2008AbovePlug(Plug):
	node : PolySmoothFace = None
	pass
class Maya65AbovePlug(Plug):
	node : PolySmoothFace = None
	pass
class MethodPlug(Plug):
	node : PolySmoothFace = None
	pass
class OrderVerticesFromFacesFirstPlug(Plug):
	node : PolySmoothFace = None
	pass
class OsdCreaseMethodPlug(Plug):
	node : PolySmoothFace = None
	pass
class OsdFvarBoundaryPlug(Plug):
	node : PolySmoothFace = None
	pass
class OsdFvarPropagateCornersPlug(Plug):
	node : PolySmoothFace = None
	pass
class OsdIndependentUVChannelsPlug(Plug):
	node : PolySmoothFace = None
	pass
class OsdSmoothTrianglesPlug(Plug):
	node : PolySmoothFace = None
	pass
class OsdVertBoundaryPlug(Plug):
	node : PolySmoothFace = None
	pass
class PropagateEdgeHardnessPlug(Plug):
	node : PolySmoothFace = None
	pass
class PushStrengthPlug(Plug):
	node : PolySmoothFace = None
	pass
class RoundnessPlug(Plug):
	node : PolySmoothFace = None
	pass
class SmoothUVsPlug(Plug):
	node : PolySmoothFace = None
	pass
class SubdivisionLevelsPlug(Plug):
	node : PolySmoothFace = None
	pass
class SubdivisionTypePlug(Plug):
	node : PolySmoothFace = None
	pass
class UseOsdBoundaryMethodsPlug(Plug):
	node : PolySmoothFace = None
	pass
# endregion


# define node class
class PolySmoothFace(PolyModifier):
	boundaryRule_ : BoundaryRulePlug = PlugDescriptor("boundaryRule")
	continuity_ : ContinuityPlug = PlugDescriptor("continuity")
	degree_ : DegreePlug = PlugDescriptor("degree")
	divisions_ : DivisionsPlug = PlugDescriptor("divisions")
	divisionsPerEdge_ : DivisionsPerEdgePlug = PlugDescriptor("divisionsPerEdge")
	keepBorder_ : KeepBorderPlug = PlugDescriptor("keepBorder")
	keepHardEdge_ : KeepHardEdgePlug = PlugDescriptor("keepHardEdge")
	keepMapBorders_ : KeepMapBordersPlug = PlugDescriptor("keepMapBorders")
	keepSelectionBorder_ : KeepSelectionBorderPlug = PlugDescriptor("keepSelectionBorder")
	keepTessellation_ : KeepTessellationPlug = PlugDescriptor("keepTessellation")
	maya2008Above_ : Maya2008AbovePlug = PlugDescriptor("maya2008Above")
	maya65Above_ : Maya65AbovePlug = PlugDescriptor("maya65Above")
	method_ : MethodPlug = PlugDescriptor("method")
	orderVerticesFromFacesFirst_ : OrderVerticesFromFacesFirstPlug = PlugDescriptor("orderVerticesFromFacesFirst")
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
	subdivisionLevels_ : SubdivisionLevelsPlug = PlugDescriptor("subdivisionLevels")
	subdivisionType_ : SubdivisionTypePlug = PlugDescriptor("subdivisionType")
	useOsdBoundaryMethods_ : UseOsdBoundaryMethodsPlug = PlugDescriptor("useOsdBoundaryMethods")

	# node attributes

	typeName = "polySmoothFace"
	typeIdInt = 1347636550
	nodeLeafClassAttrs = ["boundaryRule", "continuity", "degree", "divisions", "divisionsPerEdge", "keepBorder", "keepHardEdge", "keepMapBorders", "keepSelectionBorder", "keepTessellation", "maya2008Above", "maya65Above", "method", "orderVerticesFromFacesFirst", "osdCreaseMethod", "osdFvarBoundary", "osdFvarPropagateCorners", "osdIndependentUVChannels", "osdSmoothTriangles", "osdVertBoundary", "propagateEdgeHardness", "pushStrength", "roundness", "smoothUVs", "subdivisionLevels", "subdivisionType", "useOsdBoundaryMethods"]
	nodeLeafPlugs = ["boundaryRule", "continuity", "degree", "divisions", "divisionsPerEdge", "keepBorder", "keepHardEdge", "keepMapBorders", "keepSelectionBorder", "keepTessellation", "maya2008Above", "maya65Above", "method", "orderVerticesFromFacesFirst", "osdCreaseMethod", "osdFvarBoundary", "osdFvarPropagateCorners", "osdIndependentUVChannels", "osdSmoothTriangles", "osdVertBoundary", "propagateEdgeHardness", "pushStrength", "roundness", "smoothUVs", "subdivisionLevels", "subdivisionType", "useOsdBoundaryMethods"]
	pass

