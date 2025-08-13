

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
SurfaceShape = retriever.getNodeCls("SurfaceShape")
assert SurfaceShape
if T.TYPE_CHECKING:
	from .. import SurfaceShape

# add node doc



# region plug type defs
class BasicTessellationTypePlug(Plug):
	node : NurbsSurface = None
	pass
class CachedPlug(Plug):
	node : NurbsSurface = None
	pass
class ChordHeightPlug(Plug):
	node : NurbsSurface = None
	pass
class ChordHeightRatioPlug(Plug):
	node : NurbsSurface = None
	pass
class CreatePlug(Plug):
	node : NurbsSurface = None
	pass
class CurvatureTolerancePlug(Plug):
	node : NurbsSurface = None
	pass
class CurvePrecisionPlug(Plug):
	node : NurbsSurface = None
	pass
class CurvePrecisionShadedPlug(Plug):
	node : NurbsSurface = None
	pass
class DegreeUPlug(Plug):
	parent : DegreeUVPlug = PlugDescriptor("degreeUV")
	node : NurbsSurface = None
	pass
class DegreeVPlug(Plug):
	parent : DegreeUVPlug = PlugDescriptor("degreeUV")
	node : NurbsSurface = None
	pass
class DegreeUVPlug(Plug):
	degreeU_ : DegreeUPlug = PlugDescriptor("degreeU")
	du_ : DegreeUPlug = PlugDescriptor("degreeU")
	degreeV_ : DegreeVPlug = PlugDescriptor("degreeV")
	dv_ : DegreeVPlug = PlugDescriptor("degreeV")
	node : NurbsSurface = None
	pass
class DispCVPlug(Plug):
	node : NurbsSurface = None
	pass
class DispEPPlug(Plug):
	node : NurbsSurface = None
	pass
class DispGeometryPlug(Plug):
	node : NurbsSurface = None
	pass
class DispHullPlug(Plug):
	node : NurbsSurface = None
	pass
class DispOriginPlug(Plug):
	node : NurbsSurface = None
	pass
class DispSFPlug(Plug):
	node : NurbsSurface = None
	pass
class DisplayRenderTessellationPlug(Plug):
	node : NurbsSurface = None
	pass
class DivisionsUPlug(Plug):
	node : NurbsSurface = None
	pass
class DivisionsVPlug(Plug):
	node : NurbsSurface = None
	pass
class EdgeSwapPlug(Plug):
	node : NurbsSurface = None
	pass
class ExplicitTessellationAttributesPlug(Plug):
	node : NurbsSurface = None
	pass
class FixTextureWarpPlug(Plug):
	node : NurbsSurface = None
	pass
class FormUPlug(Plug):
	node : NurbsSurface = None
	pass
class FormVPlug(Plug):
	node : NurbsSurface = None
	pass
class GridDivisionPerSpanUPlug(Plug):
	node : NurbsSurface = None
	pass
class GridDivisionPerSpanVPlug(Plug):
	node : NurbsSurface = None
	pass
class HeaderPlug(Plug):
	node : NurbsSurface = None
	pass
class InPlacePlug(Plug):
	node : NurbsSurface = None
	pass
class LocalPlug(Plug):
	node : NurbsSurface = None
	pass
class MaxValueUPlug(Plug):
	parent : MinMaxRangeUPlug = PlugDescriptor("minMaxRangeU")
	node : NurbsSurface = None
	pass
class MinValueUPlug(Plug):
	parent : MinMaxRangeUPlug = PlugDescriptor("minMaxRangeU")
	node : NurbsSurface = None
	pass
class MinMaxRangeUPlug(Plug):
	maxValueU_ : MaxValueUPlug = PlugDescriptor("maxValueU")
	mxu_ : MaxValueUPlug = PlugDescriptor("maxValueU")
	minValueU_ : MinValueUPlug = PlugDescriptor("minValueU")
	mnu_ : MinValueUPlug = PlugDescriptor("minValueU")
	node : NurbsSurface = None
	pass
class MaxValueVPlug(Plug):
	parent : MinMaxRangeVPlug = PlugDescriptor("minMaxRangeV")
	node : NurbsSurface = None
	pass
class MinValueVPlug(Plug):
	parent : MinMaxRangeVPlug = PlugDescriptor("minMaxRangeV")
	node : NurbsSurface = None
	pass
class MinMaxRangeVPlug(Plug):
	maxValueV_ : MaxValueVPlug = PlugDescriptor("maxValueV")
	mxv_ : MaxValueVPlug = PlugDescriptor("maxValueV")
	minValueV_ : MinValueVPlug = PlugDescriptor("minValueV")
	mnv_ : MinValueVPlug = PlugDescriptor("minValueV")
	node : NurbsSurface = None
	pass
class MinScreenPlug(Plug):
	node : NurbsSurface = None
	pass
class ModeUPlug(Plug):
	node : NurbsSurface = None
	pass
class ModeVPlug(Plug):
	node : NurbsSurface = None
	pass
class NormalsDisplayScalePlug(Plug):
	node : NurbsSurface = None
	pass
class NumberUPlug(Plug):
	node : NurbsSurface = None
	pass
class NumberVPlug(Plug):
	node : NurbsSurface = None
	pass
class ObjSpaceChordHeightPlug(Plug):
	node : NurbsSurface = None
	pass
class PatchUVIdsPlug(Plug):
	node : NurbsSurface = None
	pass
class RenderTriangleCountPlug(Plug):
	node : NurbsSurface = None
	pass
class SelCVDispPlug(Plug):
	node : NurbsSurface = None
	pass
class SimplifyModePlug(Plug):
	node : NurbsSurface = None
	pass
class SimplifyUPlug(Plug):
	node : NurbsSurface = None
	pass
class SimplifyVPlug(Plug):
	node : NurbsSurface = None
	pass
class SmoothEdgePlug(Plug):
	node : NurbsSurface = None
	pass
class SmoothEdgeRatioPlug(Plug):
	node : NurbsSurface = None
	pass
class SpansUPlug(Plug):
	parent : SpansUVPlug = PlugDescriptor("spansUV")
	node : NurbsSurface = None
	pass
class SpansVPlug(Plug):
	parent : SpansUVPlug = PlugDescriptor("spansUV")
	node : NurbsSurface = None
	pass
class SpansUVPlug(Plug):
	spansU_ : SpansUPlug = PlugDescriptor("spansU")
	su_ : SpansUPlug = PlugDescriptor("spansU")
	spansV_ : SpansVPlug = PlugDescriptor("spansV")
	sv_ : SpansVPlug = PlugDescriptor("spansV")
	node : NurbsSurface = None
	pass
class TrimFacePlug(Plug):
	node : NurbsSurface = None
	pass
class TweakSizeUPlug(Plug):
	node : NurbsSurface = None
	pass
class TweakSizeVPlug(Plug):
	node : NurbsSurface = None
	pass
class UDivisionsFactorPlug(Plug):
	node : NurbsSurface = None
	pass
class UseChordHeightPlug(Plug):
	node : NurbsSurface = None
	pass
class UseChordHeightRatioPlug(Plug):
	node : NurbsSurface = None
	pass
class UseMinScreenPlug(Plug):
	node : NurbsSurface = None
	pass
class VDivisionsFactorPlug(Plug):
	node : NurbsSurface = None
	pass
class WorldSpacePlug(Plug):
	node : NurbsSurface = None
	pass
# endregion


# define node class
class NurbsSurface(SurfaceShape):
	basicTessellationType_ : BasicTessellationTypePlug = PlugDescriptor("basicTessellationType")
	cached_ : CachedPlug = PlugDescriptor("cached")
	chordHeight_ : ChordHeightPlug = PlugDescriptor("chordHeight")
	chordHeightRatio_ : ChordHeightRatioPlug = PlugDescriptor("chordHeightRatio")
	create_ : CreatePlug = PlugDescriptor("create")
	curvatureTolerance_ : CurvatureTolerancePlug = PlugDescriptor("curvatureTolerance")
	curvePrecision_ : CurvePrecisionPlug = PlugDescriptor("curvePrecision")
	curvePrecisionShaded_ : CurvePrecisionShadedPlug = PlugDescriptor("curvePrecisionShaded")
	degreeU_ : DegreeUPlug = PlugDescriptor("degreeU")
	degreeV_ : DegreeVPlug = PlugDescriptor("degreeV")
	degreeUV_ : DegreeUVPlug = PlugDescriptor("degreeUV")
	dispCV_ : DispCVPlug = PlugDescriptor("dispCV")
	dispEP_ : DispEPPlug = PlugDescriptor("dispEP")
	dispGeometry_ : DispGeometryPlug = PlugDescriptor("dispGeometry")
	dispHull_ : DispHullPlug = PlugDescriptor("dispHull")
	dispOrigin_ : DispOriginPlug = PlugDescriptor("dispOrigin")
	dispSF_ : DispSFPlug = PlugDescriptor("dispSF")
	displayRenderTessellation_ : DisplayRenderTessellationPlug = PlugDescriptor("displayRenderTessellation")
	divisionsU_ : DivisionsUPlug = PlugDescriptor("divisionsU")
	divisionsV_ : DivisionsVPlug = PlugDescriptor("divisionsV")
	edgeSwap_ : EdgeSwapPlug = PlugDescriptor("edgeSwap")
	explicitTessellationAttributes_ : ExplicitTessellationAttributesPlug = PlugDescriptor("explicitTessellationAttributes")
	fixTextureWarp_ : FixTextureWarpPlug = PlugDescriptor("fixTextureWarp")
	formU_ : FormUPlug = PlugDescriptor("formU")
	formV_ : FormVPlug = PlugDescriptor("formV")
	gridDivisionPerSpanU_ : GridDivisionPerSpanUPlug = PlugDescriptor("gridDivisionPerSpanU")
	gridDivisionPerSpanV_ : GridDivisionPerSpanVPlug = PlugDescriptor("gridDivisionPerSpanV")
	header_ : HeaderPlug = PlugDescriptor("header")
	inPlace_ : InPlacePlug = PlugDescriptor("inPlace")
	local_ : LocalPlug = PlugDescriptor("local")
	maxValueU_ : MaxValueUPlug = PlugDescriptor("maxValueU")
	minValueU_ : MinValueUPlug = PlugDescriptor("minValueU")
	minMaxRangeU_ : MinMaxRangeUPlug = PlugDescriptor("minMaxRangeU")
	maxValueV_ : MaxValueVPlug = PlugDescriptor("maxValueV")
	minValueV_ : MinValueVPlug = PlugDescriptor("minValueV")
	minMaxRangeV_ : MinMaxRangeVPlug = PlugDescriptor("minMaxRangeV")
	minScreen_ : MinScreenPlug = PlugDescriptor("minScreen")
	modeU_ : ModeUPlug = PlugDescriptor("modeU")
	modeV_ : ModeVPlug = PlugDescriptor("modeV")
	normalsDisplayScale_ : NormalsDisplayScalePlug = PlugDescriptor("normalsDisplayScale")
	numberU_ : NumberUPlug = PlugDescriptor("numberU")
	numberV_ : NumberVPlug = PlugDescriptor("numberV")
	objSpaceChordHeight_ : ObjSpaceChordHeightPlug = PlugDescriptor("objSpaceChordHeight")
	patchUVIds_ : PatchUVIdsPlug = PlugDescriptor("patchUVIds")
	renderTriangleCount_ : RenderTriangleCountPlug = PlugDescriptor("renderTriangleCount")
	selCVDisp_ : SelCVDispPlug = PlugDescriptor("selCVDisp")
	simplifyMode_ : SimplifyModePlug = PlugDescriptor("simplifyMode")
	simplifyU_ : SimplifyUPlug = PlugDescriptor("simplifyU")
	simplifyV_ : SimplifyVPlug = PlugDescriptor("simplifyV")
	smoothEdge_ : SmoothEdgePlug = PlugDescriptor("smoothEdge")
	smoothEdgeRatio_ : SmoothEdgeRatioPlug = PlugDescriptor("smoothEdgeRatio")
	spansU_ : SpansUPlug = PlugDescriptor("spansU")
	spansV_ : SpansVPlug = PlugDescriptor("spansV")
	spansUV_ : SpansUVPlug = PlugDescriptor("spansUV")
	trimFace_ : TrimFacePlug = PlugDescriptor("trimFace")
	tweakSizeU_ : TweakSizeUPlug = PlugDescriptor("tweakSizeU")
	tweakSizeV_ : TweakSizeVPlug = PlugDescriptor("tweakSizeV")
	uDivisionsFactor_ : UDivisionsFactorPlug = PlugDescriptor("uDivisionsFactor")
	useChordHeight_ : UseChordHeightPlug = PlugDescriptor("useChordHeight")
	useChordHeightRatio_ : UseChordHeightRatioPlug = PlugDescriptor("useChordHeightRatio")
	useMinScreen_ : UseMinScreenPlug = PlugDescriptor("useMinScreen")
	vDivisionsFactor_ : VDivisionsFactorPlug = PlugDescriptor("vDivisionsFactor")
	worldSpace_ : WorldSpacePlug = PlugDescriptor("worldSpace")

	# node attributes

	typeName = "nurbsSurface"
	apiTypeInt = 294
	apiTypeStr = "kNurbsSurface"
	typeIdInt = 1314083398
	MFnCls = om.MFnNurbsSurface
	pass

