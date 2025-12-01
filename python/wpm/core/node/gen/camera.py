

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Shape = retriever.getNodeCls("Shape")
assert Shape
if T.TYPE_CHECKING:
	from .. import Shape

# add node doc



# region plug type defs
class AutoSetPivotPlug(Plug):
	node : Camera = None
	pass
class BackgroundColorBPlug(Plug):
	parent : BackgroundColorPlug = PlugDescriptor("backgroundColor")
	node : Camera = None
	pass
class BackgroundColorGPlug(Plug):
	parent : BackgroundColorPlug = PlugDescriptor("backgroundColor")
	node : Camera = None
	pass
class BackgroundColorRPlug(Plug):
	parent : BackgroundColorPlug = PlugDescriptor("backgroundColor")
	node : Camera = None
	pass
class BackgroundColorPlug(Plug):
	backgroundColorB_ : BackgroundColorBPlug = PlugDescriptor("backgroundColorB")
	colb_ : BackgroundColorBPlug = PlugDescriptor("backgroundColorB")
	backgroundColorG_ : BackgroundColorGPlug = PlugDescriptor("backgroundColorG")
	colg_ : BackgroundColorGPlug = PlugDescriptor("backgroundColorG")
	backgroundColorR_ : BackgroundColorRPlug = PlugDescriptor("backgroundColorR")
	colr_ : BackgroundColorRPlug = PlugDescriptor("backgroundColorR")
	node : Camera = None
	pass
class BestFitClippingPlanesPlug(Plug):
	node : Camera = None
	pass
class BookmarksPlug(Plug):
	node : Camera = None
	pass
class BookmarksEnabledPlug(Plug):
	node : Camera = None
	pass
class HorizontalFilmAperturePlug(Plug):
	parent : CameraAperturePlug = PlugDescriptor("cameraAperture")
	node : Camera = None
	pass
class VerticalFilmAperturePlug(Plug):
	parent : CameraAperturePlug = PlugDescriptor("cameraAperture")
	node : Camera = None
	pass
class CameraAperturePlug(Plug):
	horizontalFilmAperture_ : HorizontalFilmAperturePlug = PlugDescriptor("horizontalFilmAperture")
	hfa_ : HorizontalFilmAperturePlug = PlugDescriptor("horizontalFilmAperture")
	verticalFilmAperture_ : VerticalFilmAperturePlug = PlugDescriptor("verticalFilmAperture")
	vfa_ : VerticalFilmAperturePlug = PlugDescriptor("verticalFilmAperture")
	node : Camera = None
	pass
class CameraPrecompTemplatePlug(Plug):
	node : Camera = None
	pass
class CameraScalePlug(Plug):
	node : Camera = None
	pass
class CenterOfInterestPlug(Plug):
	node : Camera = None
	pass
class ClippingPlanesPlug(Plug):
	node : Camera = None
	pass
class DepthPlug(Plug):
	node : Camera = None
	pass
class DepthNamePlug(Plug):
	node : Camera = None
	pass
class DepthOfFieldPlug(Plug):
	node : Camera = None
	pass
class DepthTypePlug(Plug):
	node : Camera = None
	pass
class DisplayCameraFarClipPlug(Plug):
	node : Camera = None
	pass
class DisplayCameraFrustumPlug(Plug):
	node : Camera = None
	pass
class DisplayCameraNearClipPlug(Plug):
	node : Camera = None
	pass
class DisplayFieldChartPlug(Plug):
	node : Camera = None
	pass
class DisplayFilmGatePlug(Plug):
	node : Camera = None
	pass
class DisplayFilmOriginPlug(Plug):
	node : Camera = None
	pass
class DisplayFilmPivotPlug(Plug):
	node : Camera = None
	pass
class DisplayGateMaskPlug(Plug):
	node : Camera = None
	pass
class DisplayGateMaskColorBPlug(Plug):
	parent : DisplayGateMaskColorPlug = PlugDescriptor("displayGateMaskColor")
	node : Camera = None
	pass
class DisplayGateMaskColorGPlug(Plug):
	parent : DisplayGateMaskColorPlug = PlugDescriptor("displayGateMaskColor")
	node : Camera = None
	pass
class DisplayGateMaskColorRPlug(Plug):
	parent : DisplayGateMaskColorPlug = PlugDescriptor("displayGateMaskColor")
	node : Camera = None
	pass
class DisplayGateMaskColorPlug(Plug):
	displayGateMaskColorB_ : DisplayGateMaskColorBPlug = PlugDescriptor("displayGateMaskColorB")
	dgcb_ : DisplayGateMaskColorBPlug = PlugDescriptor("displayGateMaskColorB")
	displayGateMaskColorG_ : DisplayGateMaskColorGPlug = PlugDescriptor("displayGateMaskColorG")
	dgcg_ : DisplayGateMaskColorGPlug = PlugDescriptor("displayGateMaskColorG")
	displayGateMaskColorR_ : DisplayGateMaskColorRPlug = PlugDescriptor("displayGateMaskColorR")
	dgcr_ : DisplayGateMaskColorRPlug = PlugDescriptor("displayGateMaskColorR")
	node : Camera = None
	pass
class DisplayGateMaskOpacityPlug(Plug):
	node : Camera = None
	pass
class DisplayResolutionPlug(Plug):
	node : Camera = None
	pass
class DisplaySafeActionPlug(Plug):
	node : Camera = None
	pass
class DisplaySafeTitlePlug(Plug):
	node : Camera = None
	pass
class FStopPlug(Plug):
	node : Camera = None
	pass
class FarClipPlanePlug(Plug):
	node : Camera = None
	pass
class FilmFitPlug(Plug):
	node : Camera = None
	pass
class FilmFitOffsetPlug(Plug):
	node : Camera = None
	pass
class HorizontalFilmOffsetPlug(Plug):
	parent : FilmOffsetPlug = PlugDescriptor("filmOffset")
	node : Camera = None
	pass
class VerticalFilmOffsetPlug(Plug):
	parent : FilmOffsetPlug = PlugDescriptor("filmOffset")
	node : Camera = None
	pass
class FilmOffsetPlug(Plug):
	horizontalFilmOffset_ : HorizontalFilmOffsetPlug = PlugDescriptor("horizontalFilmOffset")
	hfo_ : HorizontalFilmOffsetPlug = PlugDescriptor("horizontalFilmOffset")
	verticalFilmOffset_ : VerticalFilmOffsetPlug = PlugDescriptor("verticalFilmOffset")
	vfo_ : VerticalFilmOffsetPlug = PlugDescriptor("verticalFilmOffset")
	node : Camera = None
	pass
class FocalLengthPlug(Plug):
	node : Camera = None
	pass
class FocusDistancePlug(Plug):
	node : Camera = None
	pass
class FocusRegionScalePlug(Plug):
	node : Camera = None
	pass
class HomeCommandPlug(Plug):
	node : Camera = None
	pass
class ImagePlug(Plug):
	node : Camera = None
	pass
class ImageNamePlug(Plug):
	node : Camera = None
	pass
class ImagePlanePlug(Plug):
	node : Camera = None
	pass
class JournalCommandPlug(Plug):
	node : Camera = None
	pass
class LensSqueezeRatioPlug(Plug):
	node : Camera = None
	pass
class LocatorScalePlug(Plug):
	node : Camera = None
	pass
class MaskPlug(Plug):
	node : Camera = None
	pass
class MaskNamePlug(Plug):
	node : Camera = None
	pass
class MotionBlurPlug(Plug):
	node : Camera = None
	pass
class NearClipPlanePlug(Plug):
	node : Camera = None
	pass
class OrthographicPlug(Plug):
	node : Camera = None
	pass
class OrthographicWidthPlug(Plug):
	node : Camera = None
	pass
class OverscanPlug(Plug):
	node : Camera = None
	pass
class HorizontalPanPlug(Plug):
	parent : PanPlug = PlugDescriptor("pan")
	node : Camera = None
	pass
class VerticalPanPlug(Plug):
	parent : PanPlug = PlugDescriptor("pan")
	node : Camera = None
	pass
class PanPlug(Plug):
	horizontalPan_ : HorizontalPanPlug = PlugDescriptor("horizontalPan")
	hpn_ : HorizontalPanPlug = PlugDescriptor("horizontalPan")
	verticalPan_ : VerticalPanPlug = PlugDescriptor("verticalPan")
	vpn_ : VerticalPanPlug = PlugDescriptor("verticalPan")
	node : Camera = None
	pass
class PanZoomEnabledPlug(Plug):
	node : Camera = None
	pass
class FilmRollOrderPlug(Plug):
	parent : FilmRollControlPlug = PlugDescriptor("filmRollControl")
	node : Camera = None
	pass
class HorizontalRollPivotPlug(Plug):
	parent : FilmRollPivotPlug = PlugDescriptor("filmRollPivot")
	node : Camera = None
	pass
class VerticalRollPivotPlug(Plug):
	parent : FilmRollPivotPlug = PlugDescriptor("filmRollPivot")
	node : Camera = None
	pass
class FilmRollPivotPlug(Plug):
	parent : FilmRollControlPlug = PlugDescriptor("filmRollControl")
	horizontalRollPivot_ : HorizontalRollPivotPlug = PlugDescriptor("horizontalRollPivot")
	hrp_ : HorizontalRollPivotPlug = PlugDescriptor("horizontalRollPivot")
	verticalRollPivot_ : VerticalRollPivotPlug = PlugDescriptor("verticalRollPivot")
	vrp_ : VerticalRollPivotPlug = PlugDescriptor("verticalRollPivot")
	node : Camera = None
	pass
class FilmRollValuePlug(Plug):
	parent : FilmRollControlPlug = PlugDescriptor("filmRollControl")
	node : Camera = None
	pass
class FilmRollControlPlug(Plug):
	parent : PostProjectionPlug = PlugDescriptor("postProjection")
	filmRollOrder_ : FilmRollOrderPlug = PlugDescriptor("filmRollOrder")
	fro_ : FilmRollOrderPlug = PlugDescriptor("filmRollOrder")
	filmRollPivot_ : FilmRollPivotPlug = PlugDescriptor("filmRollPivot")
	frp_ : FilmRollPivotPlug = PlugDescriptor("filmRollPivot")
	filmRollValue_ : FilmRollValuePlug = PlugDescriptor("filmRollValue")
	frv_ : FilmRollValuePlug = PlugDescriptor("filmRollValue")
	node : Camera = None
	pass
class FilmTranslateHPlug(Plug):
	parent : FilmTranslatePlug = PlugDescriptor("filmTranslate")
	node : Camera = None
	pass
class FilmTranslateVPlug(Plug):
	parent : FilmTranslatePlug = PlugDescriptor("filmTranslate")
	node : Camera = None
	pass
class FilmTranslatePlug(Plug):
	parent : PostProjectionPlug = PlugDescriptor("postProjection")
	filmTranslateH_ : FilmTranslateHPlug = PlugDescriptor("filmTranslateH")
	fth_ : FilmTranslateHPlug = PlugDescriptor("filmTranslateH")
	filmTranslateV_ : FilmTranslateVPlug = PlugDescriptor("filmTranslateV")
	ftv_ : FilmTranslateVPlug = PlugDescriptor("filmTranslateV")
	node : Camera = None
	pass
class PostScalePlug(Plug):
	parent : PostProjectionPlug = PlugDescriptor("postProjection")
	node : Camera = None
	pass
class PreScalePlug(Plug):
	parent : PostProjectionPlug = PlugDescriptor("postProjection")
	node : Camera = None
	pass
class PostProjectionPlug(Plug):
	filmRollControl_ : FilmRollControlPlug = PlugDescriptor("filmRollControl")
	frc_ : FilmRollControlPlug = PlugDescriptor("filmRollControl")
	filmTranslate_ : FilmTranslatePlug = PlugDescriptor("filmTranslate")
	ct_ : FilmTranslatePlug = PlugDescriptor("filmTranslate")
	postScale_ : PostScalePlug = PlugDescriptor("postScale")
	ptsc_ : PostScalePlug = PlugDescriptor("postScale")
	preScale_ : PreScalePlug = PlugDescriptor("preScale")
	psc_ : PreScalePlug = PlugDescriptor("preScale")
	node : Camera = None
	pass
class RenderPanZoomPlug(Plug):
	node : Camera = None
	pass
class RenderablePlug(Plug):
	node : Camera = None
	pass
class HorizontalShakePlug(Plug):
	parent : ShakePlug = PlugDescriptor("shake")
	node : Camera = None
	pass
class VerticalShakePlug(Plug):
	parent : ShakePlug = PlugDescriptor("shake")
	node : Camera = None
	pass
class ShakePlug(Plug):
	horizontalShake_ : HorizontalShakePlug = PlugDescriptor("horizontalShake")
	hs_ : HorizontalShakePlug = PlugDescriptor("horizontalShake")
	verticalShake_ : VerticalShakePlug = PlugDescriptor("verticalShake")
	vs_ : VerticalShakePlug = PlugDescriptor("verticalShake")
	node : Camera = None
	pass
class ShakeEnabledPlug(Plug):
	node : Camera = None
	pass
class ShakeOverscanPlug(Plug):
	node : Camera = None
	pass
class ShakeOverscanEnabledPlug(Plug):
	node : Camera = None
	pass
class ShutterAnglePlug(Plug):
	node : Camera = None
	pass
class StereoHorizontalImageTranslatePlug(Plug):
	node : Camera = None
	pass
class StereoHorizontalImageTranslateEnabledPlug(Plug):
	node : Camera = None
	pass
class ThresholdPlug(Plug):
	node : Camera = None
	pass
class TransparencyBasedDepthPlug(Plug):
	node : Camera = None
	pass
class TriggerUpdatePlug(Plug):
	node : Camera = None
	pass
class TumblePivotXPlug(Plug):
	parent : TumblePivotPlug = PlugDescriptor("tumblePivot")
	node : Camera = None
	pass
class TumblePivotYPlug(Plug):
	parent : TumblePivotPlug = PlugDescriptor("tumblePivot")
	node : Camera = None
	pass
class TumblePivotZPlug(Plug):
	parent : TumblePivotPlug = PlugDescriptor("tumblePivot")
	node : Camera = None
	pass
class TumblePivotPlug(Plug):
	tumblePivotX_ : TumblePivotXPlug = PlugDescriptor("tumblePivotX")
	tpx_ : TumblePivotXPlug = PlugDescriptor("tumblePivotX")
	tumblePivotY_ : TumblePivotYPlug = PlugDescriptor("tumblePivotY")
	tpy_ : TumblePivotYPlug = PlugDescriptor("tumblePivotY")
	tumblePivotZ_ : TumblePivotZPlug = PlugDescriptor("tumblePivotZ")
	tpz_ : TumblePivotZPlug = PlugDescriptor("tumblePivotZ")
	node : Camera = None
	pass
class UseExploreDepthFormatPlug(Plug):
	node : Camera = None
	pass
class UsePivotAsLocalSpacePlug(Plug):
	node : Camera = None
	pass
class ZoomPlug(Plug):
	node : Camera = None
	pass
# endregion


# define node class
class Camera(Shape):
	autoSetPivot_ : AutoSetPivotPlug = PlugDescriptor("autoSetPivot")
	backgroundColorB_ : BackgroundColorBPlug = PlugDescriptor("backgroundColorB")
	backgroundColorG_ : BackgroundColorGPlug = PlugDescriptor("backgroundColorG")
	backgroundColorR_ : BackgroundColorRPlug = PlugDescriptor("backgroundColorR")
	backgroundColor_ : BackgroundColorPlug = PlugDescriptor("backgroundColor")
	bestFitClippingPlanes_ : BestFitClippingPlanesPlug = PlugDescriptor("bestFitClippingPlanes")
	bookmarks_ : BookmarksPlug = PlugDescriptor("bookmarks")
	bookmarksEnabled_ : BookmarksEnabledPlug = PlugDescriptor("bookmarksEnabled")
	horizontalFilmAperture_ : HorizontalFilmAperturePlug = PlugDescriptor("horizontalFilmAperture")
	verticalFilmAperture_ : VerticalFilmAperturePlug = PlugDescriptor("verticalFilmAperture")
	cameraAperture_ : CameraAperturePlug = PlugDescriptor("cameraAperture")
	cameraPrecompTemplate_ : CameraPrecompTemplatePlug = PlugDescriptor("cameraPrecompTemplate")
	cameraScale_ : CameraScalePlug = PlugDescriptor("cameraScale")
	centerOfInterest_ : CenterOfInterestPlug = PlugDescriptor("centerOfInterest")
	clippingPlanes_ : ClippingPlanesPlug = PlugDescriptor("clippingPlanes")
	depth_ : DepthPlug = PlugDescriptor("depth")
	depthName_ : DepthNamePlug = PlugDescriptor("depthName")
	depthOfField_ : DepthOfFieldPlug = PlugDescriptor("depthOfField")
	depthType_ : DepthTypePlug = PlugDescriptor("depthType")
	displayCameraFarClip_ : DisplayCameraFarClipPlug = PlugDescriptor("displayCameraFarClip")
	displayCameraFrustum_ : DisplayCameraFrustumPlug = PlugDescriptor("displayCameraFrustum")
	displayCameraNearClip_ : DisplayCameraNearClipPlug = PlugDescriptor("displayCameraNearClip")
	displayFieldChart_ : DisplayFieldChartPlug = PlugDescriptor("displayFieldChart")
	displayFilmGate_ : DisplayFilmGatePlug = PlugDescriptor("displayFilmGate")
	displayFilmOrigin_ : DisplayFilmOriginPlug = PlugDescriptor("displayFilmOrigin")
	displayFilmPivot_ : DisplayFilmPivotPlug = PlugDescriptor("displayFilmPivot")
	displayGateMask_ : DisplayGateMaskPlug = PlugDescriptor("displayGateMask")
	displayGateMaskColorB_ : DisplayGateMaskColorBPlug = PlugDescriptor("displayGateMaskColorB")
	displayGateMaskColorG_ : DisplayGateMaskColorGPlug = PlugDescriptor("displayGateMaskColorG")
	displayGateMaskColorR_ : DisplayGateMaskColorRPlug = PlugDescriptor("displayGateMaskColorR")
	displayGateMaskColor_ : DisplayGateMaskColorPlug = PlugDescriptor("displayGateMaskColor")
	displayGateMaskOpacity_ : DisplayGateMaskOpacityPlug = PlugDescriptor("displayGateMaskOpacity")
	displayResolution_ : DisplayResolutionPlug = PlugDescriptor("displayResolution")
	displaySafeAction_ : DisplaySafeActionPlug = PlugDescriptor("displaySafeAction")
	displaySafeTitle_ : DisplaySafeTitlePlug = PlugDescriptor("displaySafeTitle")
	fStop_ : FStopPlug = PlugDescriptor("fStop")
	farClipPlane_ : FarClipPlanePlug = PlugDescriptor("farClipPlane")
	filmFit_ : FilmFitPlug = PlugDescriptor("filmFit")
	filmFitOffset_ : FilmFitOffsetPlug = PlugDescriptor("filmFitOffset")
	horizontalFilmOffset_ : HorizontalFilmOffsetPlug = PlugDescriptor("horizontalFilmOffset")
	verticalFilmOffset_ : VerticalFilmOffsetPlug = PlugDescriptor("verticalFilmOffset")
	filmOffset_ : FilmOffsetPlug = PlugDescriptor("filmOffset")
	focalLength_ : FocalLengthPlug = PlugDescriptor("focalLength")
	focusDistance_ : FocusDistancePlug = PlugDescriptor("focusDistance")
	focusRegionScale_ : FocusRegionScalePlug = PlugDescriptor("focusRegionScale")
	homeCommand_ : HomeCommandPlug = PlugDescriptor("homeCommand")
	image_ : ImagePlug = PlugDescriptor("image")
	imageName_ : ImageNamePlug = PlugDescriptor("imageName")
	imagePlane_ : ImagePlanePlug = PlugDescriptor("imagePlane")
	journalCommand_ : JournalCommandPlug = PlugDescriptor("journalCommand")
	lensSqueezeRatio_ : LensSqueezeRatioPlug = PlugDescriptor("lensSqueezeRatio")
	locatorScale_ : LocatorScalePlug = PlugDescriptor("locatorScale")
	mask_ : MaskPlug = PlugDescriptor("mask")
	maskName_ : MaskNamePlug = PlugDescriptor("maskName")
	motionBlur_ : MotionBlurPlug = PlugDescriptor("motionBlur")
	nearClipPlane_ : NearClipPlanePlug = PlugDescriptor("nearClipPlane")
	orthographic_ : OrthographicPlug = PlugDescriptor("orthographic")
	orthographicWidth_ : OrthographicWidthPlug = PlugDescriptor("orthographicWidth")
	overscan_ : OverscanPlug = PlugDescriptor("overscan")
	horizontalPan_ : HorizontalPanPlug = PlugDescriptor("horizontalPan")
	verticalPan_ : VerticalPanPlug = PlugDescriptor("verticalPan")
	pan_ : PanPlug = PlugDescriptor("pan")
	panZoomEnabled_ : PanZoomEnabledPlug = PlugDescriptor("panZoomEnabled")
	filmRollOrder_ : FilmRollOrderPlug = PlugDescriptor("filmRollOrder")
	horizontalRollPivot_ : HorizontalRollPivotPlug = PlugDescriptor("horizontalRollPivot")
	verticalRollPivot_ : VerticalRollPivotPlug = PlugDescriptor("verticalRollPivot")
	filmRollPivot_ : FilmRollPivotPlug = PlugDescriptor("filmRollPivot")
	filmRollValue_ : FilmRollValuePlug = PlugDescriptor("filmRollValue")
	filmRollControl_ : FilmRollControlPlug = PlugDescriptor("filmRollControl")
	filmTranslateH_ : FilmTranslateHPlug = PlugDescriptor("filmTranslateH")
	filmTranslateV_ : FilmTranslateVPlug = PlugDescriptor("filmTranslateV")
	filmTranslate_ : FilmTranslatePlug = PlugDescriptor("filmTranslate")
	postScale_ : PostScalePlug = PlugDescriptor("postScale")
	preScale_ : PreScalePlug = PlugDescriptor("preScale")
	postProjection_ : PostProjectionPlug = PlugDescriptor("postProjection")
	renderPanZoom_ : RenderPanZoomPlug = PlugDescriptor("renderPanZoom")
	renderable_ : RenderablePlug = PlugDescriptor("renderable")
	horizontalShake_ : HorizontalShakePlug = PlugDescriptor("horizontalShake")
	verticalShake_ : VerticalShakePlug = PlugDescriptor("verticalShake")
	shake_ : ShakePlug = PlugDescriptor("shake")
	shakeEnabled_ : ShakeEnabledPlug = PlugDescriptor("shakeEnabled")
	shakeOverscan_ : ShakeOverscanPlug = PlugDescriptor("shakeOverscan")
	shakeOverscanEnabled_ : ShakeOverscanEnabledPlug = PlugDescriptor("shakeOverscanEnabled")
	shutterAngle_ : ShutterAnglePlug = PlugDescriptor("shutterAngle")
	stereoHorizontalImageTranslate_ : StereoHorizontalImageTranslatePlug = PlugDescriptor("stereoHorizontalImageTranslate")
	stereoHorizontalImageTranslateEnabled_ : StereoHorizontalImageTranslateEnabledPlug = PlugDescriptor("stereoHorizontalImageTranslateEnabled")
	threshold_ : ThresholdPlug = PlugDescriptor("threshold")
	transparencyBasedDepth_ : TransparencyBasedDepthPlug = PlugDescriptor("transparencyBasedDepth")
	triggerUpdate_ : TriggerUpdatePlug = PlugDescriptor("triggerUpdate")
	tumblePivotX_ : TumblePivotXPlug = PlugDescriptor("tumblePivotX")
	tumblePivotY_ : TumblePivotYPlug = PlugDescriptor("tumblePivotY")
	tumblePivotZ_ : TumblePivotZPlug = PlugDescriptor("tumblePivotZ")
	tumblePivot_ : TumblePivotPlug = PlugDescriptor("tumblePivot")
	useExploreDepthFormat_ : UseExploreDepthFormatPlug = PlugDescriptor("useExploreDepthFormat")
	usePivotAsLocalSpace_ : UsePivotAsLocalSpacePlug = PlugDescriptor("usePivotAsLocalSpace")
	zoom_ : ZoomPlug = PlugDescriptor("zoom")

	# node attributes

	typeName = "camera"
	apiTypeInt = 250
	apiTypeStr = "kCamera"
	typeIdInt = 1145258317
	MFnCls = om.MFnCamera
	nodeLeafClassAttrs = ["autoSetPivot", "backgroundColorB", "backgroundColorG", "backgroundColorR", "backgroundColor", "bestFitClippingPlanes", "bookmarks", "bookmarksEnabled", "horizontalFilmAperture", "verticalFilmAperture", "cameraAperture", "cameraPrecompTemplate", "cameraScale", "centerOfInterest", "clippingPlanes", "depth", "depthName", "depthOfField", "depthType", "displayCameraFarClip", "displayCameraFrustum", "displayCameraNearClip", "displayFieldChart", "displayFilmGate", "displayFilmOrigin", "displayFilmPivot", "displayGateMask", "displayGateMaskColorB", "displayGateMaskColorG", "displayGateMaskColorR", "displayGateMaskColor", "displayGateMaskOpacity", "displayResolution", "displaySafeAction", "displaySafeTitle", "fStop", "farClipPlane", "filmFit", "filmFitOffset", "horizontalFilmOffset", "verticalFilmOffset", "filmOffset", "focalLength", "focusDistance", "focusRegionScale", "homeCommand", "image", "imageName", "imagePlane", "journalCommand", "lensSqueezeRatio", "locatorScale", "mask", "maskName", "motionBlur", "nearClipPlane", "orthographic", "orthographicWidth", "overscan", "horizontalPan", "verticalPan", "pan", "panZoomEnabled", "filmRollOrder", "horizontalRollPivot", "verticalRollPivot", "filmRollPivot", "filmRollValue", "filmRollControl", "filmTranslateH", "filmTranslateV", "filmTranslate", "postScale", "preScale", "postProjection", "renderPanZoom", "renderable", "horizontalShake", "verticalShake", "shake", "shakeEnabled", "shakeOverscan", "shakeOverscanEnabled", "shutterAngle", "stereoHorizontalImageTranslate", "stereoHorizontalImageTranslateEnabled", "threshold", "transparencyBasedDepth", "triggerUpdate", "tumblePivotX", "tumblePivotY", "tumblePivotZ", "tumblePivot", "useExploreDepthFormat", "usePivotAsLocalSpace", "zoom"]
	nodeLeafPlugs = ["autoSetPivot", "backgroundColor", "bestFitClippingPlanes", "bookmarks", "bookmarksEnabled", "cameraAperture", "cameraPrecompTemplate", "cameraScale", "centerOfInterest", "clippingPlanes", "depth", "depthName", "depthOfField", "depthType", "displayCameraFarClip", "displayCameraFrustum", "displayCameraNearClip", "displayFieldChart", "displayFilmGate", "displayFilmOrigin", "displayFilmPivot", "displayGateMask", "displayGateMaskColor", "displayGateMaskOpacity", "displayResolution", "displaySafeAction", "displaySafeTitle", "fStop", "farClipPlane", "filmFit", "filmFitOffset", "filmOffset", "focalLength", "focusDistance", "focusRegionScale", "homeCommand", "image", "imageName", "imagePlane", "journalCommand", "lensSqueezeRatio", "locatorScale", "mask", "maskName", "motionBlur", "nearClipPlane", "orthographic", "orthographicWidth", "overscan", "pan", "panZoomEnabled", "postProjection", "renderPanZoom", "renderable", "shake", "shakeEnabled", "shakeOverscan", "shakeOverscanEnabled", "shutterAngle", "stereoHorizontalImageTranslate", "stereoHorizontalImageTranslateEnabled", "threshold", "transparencyBasedDepth", "triggerUpdate", "tumblePivot", "useExploreDepthFormat", "usePivotAsLocalSpace", "zoom"]
	pass

