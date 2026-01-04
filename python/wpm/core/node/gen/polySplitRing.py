

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifierWorld = Catalogue.PolyModifierWorld
else:
	from .. import retriever
	PolyModifierWorld = retriever.getNodeCls("PolyModifierWorld")
	assert PolyModifierWorld

# add node doc



# region plug type defs
class AbsoluteWeightPlug(Plug):
	node : PolySplitRing = None
	pass
class AdjustEdgeFlowPlug(Plug):
	node : PolySplitRing = None
	pass
class DirectionPlug(Plug):
	node : PolySplitRing = None
	pass
class DivisionsPlug(Plug):
	node : PolySplitRing = None
	pass
class EnableProfileCurvePlug(Plug):
	node : PolySplitRing = None
	pass
class FixQuadsPlug(Plug):
	node : PolySplitRing = None
	pass
class InsertWithEdgeFlowPlug(Plug):
	node : PolySplitRing = None
	pass
class ProfileCurve_FloatValuePlug(Plug):
	parent : ProfileCurvePlug = PlugDescriptor("profileCurve")
	node : PolySplitRing = None
	pass
class ProfileCurve_InterpPlug(Plug):
	parent : ProfileCurvePlug = PlugDescriptor("profileCurve")
	node : PolySplitRing = None
	pass
class ProfileCurve_PositionPlug(Plug):
	parent : ProfileCurvePlug = PlugDescriptor("profileCurve")
	node : PolySplitRing = None
	pass
class ProfileCurvePlug(Plug):
	profileCurve_FloatValue_ : ProfileCurve_FloatValuePlug = PlugDescriptor("profileCurve_FloatValue")
	pfv_ : ProfileCurve_FloatValuePlug = PlugDescriptor("profileCurve_FloatValue")
	profileCurve_Interp_ : ProfileCurve_InterpPlug = PlugDescriptor("profileCurve_Interp")
	pi_ : ProfileCurve_InterpPlug = PlugDescriptor("profileCurve_Interp")
	profileCurve_Position_ : ProfileCurve_PositionPlug = PlugDescriptor("profileCurve_Position")
	pp_ : ProfileCurve_PositionPlug = PlugDescriptor("profileCurve_Position")
	node : PolySplitRing = None
	pass
class ProfileCurveInputOffsetPlug(Plug):
	node : PolySplitRing = None
	pass
class ProfileCurveInputScalePlug(Plug):
	node : PolySplitRing = None
	pass
class RootEdgePlug(Plug):
	node : PolySplitRing = None
	pass
class SmoothingAnglePlug(Plug):
	node : PolySplitRing = None
	pass
class SplitTypePlug(Plug):
	node : PolySplitRing = None
	pass
class UseEqualMultiplierPlug(Plug):
	node : PolySplitRing = None
	pass
class UseFaceNormalsAtEndsPlug(Plug):
	node : PolySplitRing = None
	pass
class WeightPlug(Plug):
	node : PolySplitRing = None
	pass
# endregion


# define node class
class PolySplitRing(PolyModifierWorld):
	absoluteWeight_ : AbsoluteWeightPlug = PlugDescriptor("absoluteWeight")
	adjustEdgeFlow_ : AdjustEdgeFlowPlug = PlugDescriptor("adjustEdgeFlow")
	direction_ : DirectionPlug = PlugDescriptor("direction")
	divisions_ : DivisionsPlug = PlugDescriptor("divisions")
	enableProfileCurve_ : EnableProfileCurvePlug = PlugDescriptor("enableProfileCurve")
	fixQuads_ : FixQuadsPlug = PlugDescriptor("fixQuads")
	insertWithEdgeFlow_ : InsertWithEdgeFlowPlug = PlugDescriptor("insertWithEdgeFlow")
	profileCurve_FloatValue_ : ProfileCurve_FloatValuePlug = PlugDescriptor("profileCurve_FloatValue")
	profileCurve_Interp_ : ProfileCurve_InterpPlug = PlugDescriptor("profileCurve_Interp")
	profileCurve_Position_ : ProfileCurve_PositionPlug = PlugDescriptor("profileCurve_Position")
	profileCurve_ : ProfileCurvePlug = PlugDescriptor("profileCurve")
	profileCurveInputOffset_ : ProfileCurveInputOffsetPlug = PlugDescriptor("profileCurveInputOffset")
	profileCurveInputScale_ : ProfileCurveInputScalePlug = PlugDescriptor("profileCurveInputScale")
	rootEdge_ : RootEdgePlug = PlugDescriptor("rootEdge")
	smoothingAngle_ : SmoothingAnglePlug = PlugDescriptor("smoothingAngle")
	splitType_ : SplitTypePlug = PlugDescriptor("splitType")
	useEqualMultiplier_ : UseEqualMultiplierPlug = PlugDescriptor("useEqualMultiplier")
	useFaceNormalsAtEnds_ : UseFaceNormalsAtEndsPlug = PlugDescriptor("useFaceNormalsAtEnds")
	weight_ : WeightPlug = PlugDescriptor("weight")

	# node attributes

	typeName = "polySplitRing"
	apiTypeInt = 970
	apiTypeStr = "kPolySplitRing"
	typeIdInt = 1347637330
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["absoluteWeight", "adjustEdgeFlow", "direction", "divisions", "enableProfileCurve", "fixQuads", "insertWithEdgeFlow", "profileCurve_FloatValue", "profileCurve_Interp", "profileCurve_Position", "profileCurve", "profileCurveInputOffset", "profileCurveInputScale", "rootEdge", "smoothingAngle", "splitType", "useEqualMultiplier", "useFaceNormalsAtEnds", "weight"]
	nodeLeafPlugs = ["absoluteWeight", "adjustEdgeFlow", "direction", "divisions", "enableProfileCurve", "fixQuads", "insertWithEdgeFlow", "profileCurve", "profileCurveInputOffset", "profileCurveInputScale", "rootEdge", "smoothingAngle", "splitType", "useEqualMultiplier", "useFaceNormalsAtEnds", "weight"]
	pass

