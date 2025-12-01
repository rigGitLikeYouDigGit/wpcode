

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
assert AbstractBaseCreate
if T.TYPE_CHECKING:
	from .. import AbstractBaseCreate

# add node doc



# region plug type defs
class BevelInsidePlug(Plug):
	node : BevelPlus = None
	pass
class CapSidesPlug(Plug):
	node : BevelPlus = None
	pass
class CountPlug(Plug):
	node : BevelPlus = None
	pass
class DepthPlug(Plug):
	node : BevelPlus = None
	pass
class EndCapSurfacePlug(Plug):
	node : BevelPlus = None
	pass
class ExtrudeDepthPlug(Plug):
	node : BevelPlus = None
	pass
class InnerStyleCurvePlug(Plug):
	node : BevelPlus = None
	pass
class InputCurvesPlug(Plug):
	node : BevelPlus = None
	pass
class JoinSurfacesPlug(Plug):
	node : BevelPlus = None
	pass
class NormalsOutwardsPlug(Plug):
	node : BevelPlus = None
	pass
class NumberOfSidesPlug(Plug):
	node : BevelPlus = None
	pass
class OrderedCurvesPlug(Plug):
	node : BevelPlus = None
	pass
class OuterStyleCurvePlug(Plug):
	node : BevelPlus = None
	pass
class OutputPolyPlug(Plug):
	node : BevelPlus = None
	pass
class OutputSurfacesPlug(Plug):
	node : BevelPlus = None
	pass
class PolyOutChordHeightPlug(Plug):
	node : BevelPlus = None
	pass
class PolyOutChordHeightRatioPlug(Plug):
	node : BevelPlus = None
	pass
class PolyOutCountPlug(Plug):
	node : BevelPlus = None
	pass
class PolyOutCurveSamplesPlug(Plug):
	node : BevelPlus = None
	pass
class PolyOutCurveTypePlug(Plug):
	node : BevelPlus = None
	pass
class PolyOutExtrusionSamplesPlug(Plug):
	node : BevelPlus = None
	pass
class PolyOutExtrusionTypePlug(Plug):
	node : BevelPlus = None
	pass
class PolyOutMethodPlug(Plug):
	node : BevelPlus = None
	pass
class PolyOutUseChordHeightPlug(Plug):
	node : BevelPlus = None
	pass
class PolyOutUseChordHeightRatioPlug(Plug):
	node : BevelPlus = None
	pass
class PositionXPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : BevelPlus = None
	pass
class PositionYPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : BevelPlus = None
	pass
class PositionZPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : BevelPlus = None
	pass
class PositionPlug(Plug):
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	px_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	py_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	pz_ : PositionZPlug = PlugDescriptor("positionZ")
	node : BevelPlus = None
	pass
class StartCapSurfacePlug(Plug):
	node : BevelPlus = None
	pass
class TolerancePlug(Plug):
	node : BevelPlus = None
	pass
class WidthPlug(Plug):
	node : BevelPlus = None
	pass
# endregion


# define node class
class BevelPlus(AbstractBaseCreate):
	bevelInside_ : BevelInsidePlug = PlugDescriptor("bevelInside")
	capSides_ : CapSidesPlug = PlugDescriptor("capSides")
	count_ : CountPlug = PlugDescriptor("count")
	depth_ : DepthPlug = PlugDescriptor("depth")
	endCapSurface_ : EndCapSurfacePlug = PlugDescriptor("endCapSurface")
	extrudeDepth_ : ExtrudeDepthPlug = PlugDescriptor("extrudeDepth")
	innerStyleCurve_ : InnerStyleCurvePlug = PlugDescriptor("innerStyleCurve")
	inputCurves_ : InputCurvesPlug = PlugDescriptor("inputCurves")
	joinSurfaces_ : JoinSurfacesPlug = PlugDescriptor("joinSurfaces")
	normalsOutwards_ : NormalsOutwardsPlug = PlugDescriptor("normalsOutwards")
	numberOfSides_ : NumberOfSidesPlug = PlugDescriptor("numberOfSides")
	orderedCurves_ : OrderedCurvesPlug = PlugDescriptor("orderedCurves")
	outerStyleCurve_ : OuterStyleCurvePlug = PlugDescriptor("outerStyleCurve")
	outputPoly_ : OutputPolyPlug = PlugDescriptor("outputPoly")
	outputSurfaces_ : OutputSurfacesPlug = PlugDescriptor("outputSurfaces")
	polyOutChordHeight_ : PolyOutChordHeightPlug = PlugDescriptor("polyOutChordHeight")
	polyOutChordHeightRatio_ : PolyOutChordHeightRatioPlug = PlugDescriptor("polyOutChordHeightRatio")
	polyOutCount_ : PolyOutCountPlug = PlugDescriptor("polyOutCount")
	polyOutCurveSamples_ : PolyOutCurveSamplesPlug = PlugDescriptor("polyOutCurveSamples")
	polyOutCurveType_ : PolyOutCurveTypePlug = PlugDescriptor("polyOutCurveType")
	polyOutExtrusionSamples_ : PolyOutExtrusionSamplesPlug = PlugDescriptor("polyOutExtrusionSamples")
	polyOutExtrusionType_ : PolyOutExtrusionTypePlug = PlugDescriptor("polyOutExtrusionType")
	polyOutMethod_ : PolyOutMethodPlug = PlugDescriptor("polyOutMethod")
	polyOutUseChordHeight_ : PolyOutUseChordHeightPlug = PlugDescriptor("polyOutUseChordHeight")
	polyOutUseChordHeightRatio_ : PolyOutUseChordHeightRatioPlug = PlugDescriptor("polyOutUseChordHeightRatio")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	position_ : PositionPlug = PlugDescriptor("position")
	startCapSurface_ : StartCapSurfacePlug = PlugDescriptor("startCapSurface")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")
	width_ : WidthPlug = PlugDescriptor("width")

	# node attributes

	typeName = "bevelPlus"
	apiTypeInt = 899
	apiTypeStr = "kBevelPlus"
	typeIdInt = 1312969558
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["bevelInside", "capSides", "count", "depth", "endCapSurface", "extrudeDepth", "innerStyleCurve", "inputCurves", "joinSurfaces", "normalsOutwards", "numberOfSides", "orderedCurves", "outerStyleCurve", "outputPoly", "outputSurfaces", "polyOutChordHeight", "polyOutChordHeightRatio", "polyOutCount", "polyOutCurveSamples", "polyOutCurveType", "polyOutExtrusionSamples", "polyOutExtrusionType", "polyOutMethod", "polyOutUseChordHeight", "polyOutUseChordHeightRatio", "positionX", "positionY", "positionZ", "position", "startCapSurface", "tolerance", "width"]
	nodeLeafPlugs = ["bevelInside", "capSides", "count", "depth", "endCapSurface", "extrudeDepth", "innerStyleCurve", "inputCurves", "joinSurfaces", "normalsOutwards", "numberOfSides", "orderedCurves", "outerStyleCurve", "outputPoly", "outputSurfaces", "polyOutChordHeight", "polyOutChordHeightRatio", "polyOutCount", "polyOutCurveSamples", "polyOutCurveType", "polyOutExtrusionSamples", "polyOutExtrusionType", "polyOutMethod", "polyOutUseChordHeight", "polyOutUseChordHeightRatio", "position", "startCapSurface", "tolerance", "width"]
	pass

