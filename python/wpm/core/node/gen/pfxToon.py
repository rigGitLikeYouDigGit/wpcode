

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PfxGeometry = retriever.getNodeCls("PfxGeometry")
assert PfxGeometry
if T.TYPE_CHECKING:
	from .. import PfxGeometry

# add node doc



# region plug type defs
class BackfacingCreasesPlug(Plug):
	node : PfxToon = None
	pass
class BorderBreakAnglePlug(Plug):
	node : PfxToon = None
	pass
class BorderColorBPlug(Plug):
	parent : BorderColorPlug = PlugDescriptor("borderColor")
	node : PfxToon = None
	pass
class BorderColorGPlug(Plug):
	parent : BorderColorPlug = PlugDescriptor("borderColor")
	node : PfxToon = None
	pass
class BorderColorRPlug(Plug):
	parent : BorderColorPlug = PlugDescriptor("borderColor")
	node : PfxToon = None
	pass
class BorderColorPlug(Plug):
	borderColorB_ : BorderColorBPlug = PlugDescriptor("borderColorB")
	bcb_ : BorderColorBPlug = PlugDescriptor("borderColorB")
	borderColorG_ : BorderColorGPlug = PlugDescriptor("borderColorG")
	bcg_ : BorderColorGPlug = PlugDescriptor("borderColorG")
	borderColorR_ : BorderColorRPlug = PlugDescriptor("borderColorR")
	bcr_ : BorderColorRPlug = PlugDescriptor("borderColorR")
	node : PfxToon = None
	pass
class BorderLineWidthPlug(Plug):
	node : PfxToon = None
	pass
class BorderLinesPlug(Plug):
	node : PfxToon = None
	pass
class BorderWidthModulationPlug(Plug):
	node : PfxToon = None
	pass
class CreaseAngleMaxPlug(Plug):
	node : PfxToon = None
	pass
class CreaseAngleMinPlug(Plug):
	node : PfxToon = None
	pass
class CreaseBreakAnglePlug(Plug):
	node : PfxToon = None
	pass
class CreaseColorBPlug(Plug):
	parent : CreaseColorPlug = PlugDescriptor("creaseColor")
	node : PfxToon = None
	pass
class CreaseColorGPlug(Plug):
	parent : CreaseColorPlug = PlugDescriptor("creaseColor")
	node : PfxToon = None
	pass
class CreaseColorRPlug(Plug):
	parent : CreaseColorPlug = PlugDescriptor("creaseColor")
	node : PfxToon = None
	pass
class CreaseColorPlug(Plug):
	creaseColorB_ : CreaseColorBPlug = PlugDescriptor("creaseColorB")
	ccb_ : CreaseColorBPlug = PlugDescriptor("creaseColorB")
	creaseColorG_ : CreaseColorGPlug = PlugDescriptor("creaseColorG")
	ccg_ : CreaseColorGPlug = PlugDescriptor("creaseColorG")
	creaseColorR_ : CreaseColorRPlug = PlugDescriptor("creaseColorR")
	ccr_ : CreaseColorRPlug = PlugDescriptor("creaseColorR")
	node : PfxToon = None
	pass
class CreaseLineWidthPlug(Plug):
	node : PfxToon = None
	pass
class CreaseLinesPlug(Plug):
	node : PfxToon = None
	pass
class CreaseWidthModulationPlug(Plug):
	node : PfxToon = None
	pass
class CurvatureModulationPlug(Plug):
	node : PfxToon = None
	pass
class CurvatureWidth_FloatValuePlug(Plug):
	parent : CurvatureWidthPlug = PlugDescriptor("curvatureWidth")
	node : PfxToon = None
	pass
class CurvatureWidth_InterpPlug(Plug):
	parent : CurvatureWidthPlug = PlugDescriptor("curvatureWidth")
	node : PfxToon = None
	pass
class CurvatureWidth_PositionPlug(Plug):
	parent : CurvatureWidthPlug = PlugDescriptor("curvatureWidth")
	node : PfxToon = None
	pass
class CurvatureWidthPlug(Plug):
	curvatureWidth_FloatValue_ : CurvatureWidth_FloatValuePlug = PlugDescriptor("curvatureWidth_FloatValue")
	cwdfv_ : CurvatureWidth_FloatValuePlug = PlugDescriptor("curvatureWidth_FloatValue")
	curvatureWidth_Interp_ : CurvatureWidth_InterpPlug = PlugDescriptor("curvatureWidth_Interp")
	cwdi_ : CurvatureWidth_InterpPlug = PlugDescriptor("curvatureWidth_Interp")
	curvatureWidth_Position_ : CurvatureWidth_PositionPlug = PlugDescriptor("curvatureWidth_Position")
	cwdp_ : CurvatureWidth_PositionPlug = PlugDescriptor("curvatureWidth_Position")
	node : PfxToon = None
	pass
class DepthBiasPlug(Plug):
	node : PfxToon = None
	pass
class DepthOffsetPlug(Plug):
	node : PfxToon = None
	pass
class DisplayInViewportPlug(Plug):
	node : PfxToon = None
	pass
class DistanceScalingPlug(Plug):
	node : PfxToon = None
	pass
class FlushAngleMaxPlug(Plug):
	node : PfxToon = None
	pass
class FlushTolerancePlug(Plug):
	node : PfxToon = None
	pass
class HardCreasesOnlyPlug(Plug):
	node : PfxToon = None
	pass
class InputWorldMatrixPlug(Plug):
	parent : InputSurfacePlug = PlugDescriptor("inputSurface")
	node : PfxToon = None
	pass
class SurfacePlug(Plug):
	parent : InputSurfacePlug = PlugDescriptor("inputSurface")
	node : PfxToon = None
	pass
class InputSurfacePlug(Plug):
	inputWorldMatrix_ : InputWorldMatrixPlug = PlugDescriptor("inputWorldMatrix")
	iwm_ : InputWorldMatrixPlug = PlugDescriptor("inputWorldMatrix")
	surface_ : SurfacePlug = PlugDescriptor("surface")
	srf_ : SurfacePlug = PlugDescriptor("surface")
	node : PfxToon = None
	pass
class IntersectionAngleMaxPlug(Plug):
	node : PfxToon = None
	pass
class IntersectionAngleMinPlug(Plug):
	node : PfxToon = None
	pass
class IntersectionBreakAnglePlug(Plug):
	node : PfxToon = None
	pass
class IntersectionColorBPlug(Plug):
	parent : IntersectionColorPlug = PlugDescriptor("intersectionColor")
	node : PfxToon = None
	pass
class IntersectionColorGPlug(Plug):
	parent : IntersectionColorPlug = PlugDescriptor("intersectionColor")
	node : PfxToon = None
	pass
class IntersectionColorRPlug(Plug):
	parent : IntersectionColorPlug = PlugDescriptor("intersectionColor")
	node : PfxToon = None
	pass
class IntersectionColorPlug(Plug):
	intersectionColorB_ : IntersectionColorBPlug = PlugDescriptor("intersectionColorB")
	icb_ : IntersectionColorBPlug = PlugDescriptor("intersectionColorB")
	intersectionColorG_ : IntersectionColorGPlug = PlugDescriptor("intersectionColorG")
	icg_ : IntersectionColorGPlug = PlugDescriptor("intersectionColorG")
	intersectionColorR_ : IntersectionColorRPlug = PlugDescriptor("intersectionColorR")
	icr_ : IntersectionColorRPlug = PlugDescriptor("intersectionColorR")
	node : PfxToon = None
	pass
class IntersectionLineWidthPlug(Plug):
	node : PfxToon = None
	pass
class IntersectionLinesPlug(Plug):
	node : PfxToon = None
	pass
class IntersectionWidthModulationPlug(Plug):
	node : PfxToon = None
	pass
class LightingBasedWidthPlug(Plug):
	node : PfxToon = None
	pass
class LineEndThinningPlug(Plug):
	node : PfxToon = None
	pass
class LineExtendPlug(Plug):
	node : PfxToon = None
	pass
class LineOffsetPlug(Plug):
	node : PfxToon = None
	pass
class LineOffsetMapPlug(Plug):
	node : PfxToon = None
	pass
class LineOpacityPlug(Plug):
	node : PfxToon = None
	pass
class LineOpacityMapPlug(Plug):
	node : PfxToon = None
	pass
class LineWidthPlug(Plug):
	node : PfxToon = None
	pass
class LineWidthMapPlug(Plug):
	node : PfxToon = None
	pass
class LocalOcclusionPlug(Plug):
	node : PfxToon = None
	pass
class MaxPixelWidthPlug(Plug):
	node : PfxToon = None
	pass
class MaxSegmentLengthPlug(Plug):
	node : PfxToon = None
	pass
class MinPixelWidthPlug(Plug):
	node : PfxToon = None
	pass
class MinSegmentLengthPlug(Plug):
	node : PfxToon = None
	pass
class OcclusionTolerancePlug(Plug):
	node : PfxToon = None
	pass
class OcclusionWidthScalePlug(Plug):
	node : PfxToon = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : PfxToon = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : PfxToon = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : PfxToon = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : PfxToon = None
	pass
class OutProfileMeshPlug(Plug):
	node : PfxToon = None
	pass
class PfxRandomizePlug(Plug):
	node : PfxToon = None
	pass
class ProfileBreakAnglePlug(Plug):
	node : PfxToon = None
	pass
class ProfileColorBPlug(Plug):
	parent : ProfileColorPlug = PlugDescriptor("profileColor")
	node : PfxToon = None
	pass
class ProfileColorGPlug(Plug):
	parent : ProfileColorPlug = PlugDescriptor("profileColor")
	node : PfxToon = None
	pass
class ProfileColorRPlug(Plug):
	parent : ProfileColorPlug = PlugDescriptor("profileColor")
	node : PfxToon = None
	pass
class ProfileColorPlug(Plug):
	profileColorB_ : ProfileColorBPlug = PlugDescriptor("profileColorB")
	pcb_ : ProfileColorBPlug = PlugDescriptor("profileColorB")
	profileColorG_ : ProfileColorGPlug = PlugDescriptor("profileColorG")
	pcg_ : ProfileColorGPlug = PlugDescriptor("profileColorG")
	profileColorR_ : ProfileColorRPlug = PlugDescriptor("profileColorR")
	pcr_ : ProfileColorRPlug = PlugDescriptor("profileColorR")
	node : PfxToon = None
	pass
class ProfileLineWidthPlug(Plug):
	node : PfxToon = None
	pass
class ProfileLinesPlug(Plug):
	node : PfxToon = None
	pass
class ProfileWidthModulationPlug(Plug):
	node : PfxToon = None
	pass
class RemoveFlushBordersPlug(Plug):
	node : PfxToon = None
	pass
class ResampleBorderPlug(Plug):
	node : PfxToon = None
	pass
class ResampleCreasePlug(Plug):
	node : PfxToon = None
	pass
class ResampleIntersectionPlug(Plug):
	node : PfxToon = None
	pass
class ResampleProfilePlug(Plug):
	node : PfxToon = None
	pass
class ScreenSpaceResamplingPlug(Plug):
	node : PfxToon = None
	pass
class ScreenspaceWidthPlug(Plug):
	node : PfxToon = None
	pass
class SelfIntersectPlug(Plug):
	node : PfxToon = None
	pass
class SmoothProfilePlug(Plug):
	node : PfxToon = None
	pass
class TighterProfilePlug(Plug):
	node : PfxToon = None
	pass
# endregion


# define node class
class PfxToon(PfxGeometry):
	backfacingCreases_ : BackfacingCreasesPlug = PlugDescriptor("backfacingCreases")
	borderBreakAngle_ : BorderBreakAnglePlug = PlugDescriptor("borderBreakAngle")
	borderColorB_ : BorderColorBPlug = PlugDescriptor("borderColorB")
	borderColorG_ : BorderColorGPlug = PlugDescriptor("borderColorG")
	borderColorR_ : BorderColorRPlug = PlugDescriptor("borderColorR")
	borderColor_ : BorderColorPlug = PlugDescriptor("borderColor")
	borderLineWidth_ : BorderLineWidthPlug = PlugDescriptor("borderLineWidth")
	borderLines_ : BorderLinesPlug = PlugDescriptor("borderLines")
	borderWidthModulation_ : BorderWidthModulationPlug = PlugDescriptor("borderWidthModulation")
	creaseAngleMax_ : CreaseAngleMaxPlug = PlugDescriptor("creaseAngleMax")
	creaseAngleMin_ : CreaseAngleMinPlug = PlugDescriptor("creaseAngleMin")
	creaseBreakAngle_ : CreaseBreakAnglePlug = PlugDescriptor("creaseBreakAngle")
	creaseColorB_ : CreaseColorBPlug = PlugDescriptor("creaseColorB")
	creaseColorG_ : CreaseColorGPlug = PlugDescriptor("creaseColorG")
	creaseColorR_ : CreaseColorRPlug = PlugDescriptor("creaseColorR")
	creaseColor_ : CreaseColorPlug = PlugDescriptor("creaseColor")
	creaseLineWidth_ : CreaseLineWidthPlug = PlugDescriptor("creaseLineWidth")
	creaseLines_ : CreaseLinesPlug = PlugDescriptor("creaseLines")
	creaseWidthModulation_ : CreaseWidthModulationPlug = PlugDescriptor("creaseWidthModulation")
	curvatureModulation_ : CurvatureModulationPlug = PlugDescriptor("curvatureModulation")
	curvatureWidth_FloatValue_ : CurvatureWidth_FloatValuePlug = PlugDescriptor("curvatureWidth_FloatValue")
	curvatureWidth_Interp_ : CurvatureWidth_InterpPlug = PlugDescriptor("curvatureWidth_Interp")
	curvatureWidth_Position_ : CurvatureWidth_PositionPlug = PlugDescriptor("curvatureWidth_Position")
	curvatureWidth_ : CurvatureWidthPlug = PlugDescriptor("curvatureWidth")
	depthBias_ : DepthBiasPlug = PlugDescriptor("depthBias")
	depthOffset_ : DepthOffsetPlug = PlugDescriptor("depthOffset")
	displayInViewport_ : DisplayInViewportPlug = PlugDescriptor("displayInViewport")
	distanceScaling_ : DistanceScalingPlug = PlugDescriptor("distanceScaling")
	flushAngleMax_ : FlushAngleMaxPlug = PlugDescriptor("flushAngleMax")
	flushTolerance_ : FlushTolerancePlug = PlugDescriptor("flushTolerance")
	hardCreasesOnly_ : HardCreasesOnlyPlug = PlugDescriptor("hardCreasesOnly")
	inputWorldMatrix_ : InputWorldMatrixPlug = PlugDescriptor("inputWorldMatrix")
	surface_ : SurfacePlug = PlugDescriptor("surface")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	intersectionAngleMax_ : IntersectionAngleMaxPlug = PlugDescriptor("intersectionAngleMax")
	intersectionAngleMin_ : IntersectionAngleMinPlug = PlugDescriptor("intersectionAngleMin")
	intersectionBreakAngle_ : IntersectionBreakAnglePlug = PlugDescriptor("intersectionBreakAngle")
	intersectionColorB_ : IntersectionColorBPlug = PlugDescriptor("intersectionColorB")
	intersectionColorG_ : IntersectionColorGPlug = PlugDescriptor("intersectionColorG")
	intersectionColorR_ : IntersectionColorRPlug = PlugDescriptor("intersectionColorR")
	intersectionColor_ : IntersectionColorPlug = PlugDescriptor("intersectionColor")
	intersectionLineWidth_ : IntersectionLineWidthPlug = PlugDescriptor("intersectionLineWidth")
	intersectionLines_ : IntersectionLinesPlug = PlugDescriptor("intersectionLines")
	intersectionWidthModulation_ : IntersectionWidthModulationPlug = PlugDescriptor("intersectionWidthModulation")
	lightingBasedWidth_ : LightingBasedWidthPlug = PlugDescriptor("lightingBasedWidth")
	lineEndThinning_ : LineEndThinningPlug = PlugDescriptor("lineEndThinning")
	lineExtend_ : LineExtendPlug = PlugDescriptor("lineExtend")
	lineOffset_ : LineOffsetPlug = PlugDescriptor("lineOffset")
	lineOffsetMap_ : LineOffsetMapPlug = PlugDescriptor("lineOffsetMap")
	lineOpacity_ : LineOpacityPlug = PlugDescriptor("lineOpacity")
	lineOpacityMap_ : LineOpacityMapPlug = PlugDescriptor("lineOpacityMap")
	lineWidth_ : LineWidthPlug = PlugDescriptor("lineWidth")
	lineWidthMap_ : LineWidthMapPlug = PlugDescriptor("lineWidthMap")
	localOcclusion_ : LocalOcclusionPlug = PlugDescriptor("localOcclusion")
	maxPixelWidth_ : MaxPixelWidthPlug = PlugDescriptor("maxPixelWidth")
	maxSegmentLength_ : MaxSegmentLengthPlug = PlugDescriptor("maxSegmentLength")
	minPixelWidth_ : MinPixelWidthPlug = PlugDescriptor("minPixelWidth")
	minSegmentLength_ : MinSegmentLengthPlug = PlugDescriptor("minSegmentLength")
	occlusionTolerance_ : OcclusionTolerancePlug = PlugDescriptor("occlusionTolerance")
	occlusionWidthScale_ : OcclusionWidthScalePlug = PlugDescriptor("occlusionWidthScale")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	outProfileMesh_ : OutProfileMeshPlug = PlugDescriptor("outProfileMesh")
	pfxRandomize_ : PfxRandomizePlug = PlugDescriptor("pfxRandomize")
	profileBreakAngle_ : ProfileBreakAnglePlug = PlugDescriptor("profileBreakAngle")
	profileColorB_ : ProfileColorBPlug = PlugDescriptor("profileColorB")
	profileColorG_ : ProfileColorGPlug = PlugDescriptor("profileColorG")
	profileColorR_ : ProfileColorRPlug = PlugDescriptor("profileColorR")
	profileColor_ : ProfileColorPlug = PlugDescriptor("profileColor")
	profileLineWidth_ : ProfileLineWidthPlug = PlugDescriptor("profileLineWidth")
	profileLines_ : ProfileLinesPlug = PlugDescriptor("profileLines")
	profileWidthModulation_ : ProfileWidthModulationPlug = PlugDescriptor("profileWidthModulation")
	removeFlushBorders_ : RemoveFlushBordersPlug = PlugDescriptor("removeFlushBorders")
	resampleBorder_ : ResampleBorderPlug = PlugDescriptor("resampleBorder")
	resampleCrease_ : ResampleCreasePlug = PlugDescriptor("resampleCrease")
	resampleIntersection_ : ResampleIntersectionPlug = PlugDescriptor("resampleIntersection")
	resampleProfile_ : ResampleProfilePlug = PlugDescriptor("resampleProfile")
	screenSpaceResampling_ : ScreenSpaceResamplingPlug = PlugDescriptor("screenSpaceResampling")
	screenspaceWidth_ : ScreenspaceWidthPlug = PlugDescriptor("screenspaceWidth")
	selfIntersect_ : SelfIntersectPlug = PlugDescriptor("selfIntersect")
	smoothProfile_ : SmoothProfilePlug = PlugDescriptor("smoothProfile")
	tighterProfile_ : TighterProfilePlug = PlugDescriptor("tighterProfile")

	# node attributes

	typeName = "pfxToon"
	apiTypeInt = 971
	apiTypeStr = "kPfxToon"
	typeIdInt = 1346786383
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["backfacingCreases", "borderBreakAngle", "borderColorB", "borderColorG", "borderColorR", "borderColor", "borderLineWidth", "borderLines", "borderWidthModulation", "creaseAngleMax", "creaseAngleMin", "creaseBreakAngle", "creaseColorB", "creaseColorG", "creaseColorR", "creaseColor", "creaseLineWidth", "creaseLines", "creaseWidthModulation", "curvatureModulation", "curvatureWidth_FloatValue", "curvatureWidth_Interp", "curvatureWidth_Position", "curvatureWidth", "depthBias", "depthOffset", "displayInViewport", "distanceScaling", "flushAngleMax", "flushTolerance", "hardCreasesOnly", "inputWorldMatrix", "surface", "inputSurface", "intersectionAngleMax", "intersectionAngleMin", "intersectionBreakAngle", "intersectionColorB", "intersectionColorG", "intersectionColorR", "intersectionColor", "intersectionLineWidth", "intersectionLines", "intersectionWidthModulation", "lightingBasedWidth", "lineEndThinning", "lineExtend", "lineOffset", "lineOffsetMap", "lineOpacity", "lineOpacityMap", "lineWidth", "lineWidthMap", "localOcclusion", "maxPixelWidth", "maxSegmentLength", "minPixelWidth", "minSegmentLength", "occlusionTolerance", "occlusionWidthScale", "outColorB", "outColorG", "outColorR", "outColor", "outProfileMesh", "pfxRandomize", "profileBreakAngle", "profileColorB", "profileColorG", "profileColorR", "profileColor", "profileLineWidth", "profileLines", "profileWidthModulation", "removeFlushBorders", "resampleBorder", "resampleCrease", "resampleIntersection", "resampleProfile", "screenSpaceResampling", "screenspaceWidth", "selfIntersect", "smoothProfile", "tighterProfile"]
	nodeLeafPlugs = ["backfacingCreases", "borderBreakAngle", "borderColor", "borderLineWidth", "borderLines", "borderWidthModulation", "creaseAngleMax", "creaseAngleMin", "creaseBreakAngle", "creaseColor", "creaseLineWidth", "creaseLines", "creaseWidthModulation", "curvatureModulation", "curvatureWidth", "depthBias", "depthOffset", "displayInViewport", "distanceScaling", "flushAngleMax", "flushTolerance", "hardCreasesOnly", "inputSurface", "intersectionAngleMax", "intersectionAngleMin", "intersectionBreakAngle", "intersectionColor", "intersectionLineWidth", "intersectionLines", "intersectionWidthModulation", "lightingBasedWidth", "lineEndThinning", "lineExtend", "lineOffset", "lineOffsetMap", "lineOpacity", "lineOpacityMap", "lineWidth", "lineWidthMap", "localOcclusion", "maxPixelWidth", "maxSegmentLength", "minPixelWidth", "minSegmentLength", "occlusionTolerance", "occlusionWidthScale", "outColor", "outProfileMesh", "pfxRandomize", "profileBreakAngle", "profileColor", "profileLineWidth", "profileLines", "profileWidthModulation", "removeFlushBorders", "resampleBorder", "resampleCrease", "resampleIntersection", "resampleProfile", "screenSpaceResampling", "screenspaceWidth", "selfIntersect", "smoothProfile", "tighterProfile"]
	pass

