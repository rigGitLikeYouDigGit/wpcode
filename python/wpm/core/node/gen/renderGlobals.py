

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class AnimationPlug(Plug):
	node : RenderGlobals = None
	pass
class AnimationRangePlug(Plug):
	node : RenderGlobals = None
	pass
class ApplyFogInPostPlug(Plug):
	node : RenderGlobals = None
	pass
class BinMembershipPlug(Plug):
	node : RenderGlobals = None
	pass
class BitDepthPlug(Plug):
	node : RenderGlobals = None
	pass
class Blur2DMemoryCapPlug(Plug):
	node : RenderGlobals = None
	pass
class BlurLengthPlug(Plug):
	node : RenderGlobals = None
	pass
class BlurSharpnessPlug(Plug):
	node : RenderGlobals = None
	pass
class BottomRegionPlug(Plug):
	node : RenderGlobals = None
	pass
class BufferNamePlug(Plug):
	node : RenderGlobals = None
	pass
class ByExtensionPlug(Plug):
	node : RenderGlobals = None
	pass
class ByFramePlug(Plug):
	node : RenderGlobals = None
	pass
class ByFrameStepPlug(Plug):
	node : RenderGlobals = None
	pass
class ClipFinalShadedColorPlug(Plug):
	node : RenderGlobals = None
	pass
class ColorProfileEnabledPlug(Plug):
	node : RenderGlobals = None
	pass
class ComFrrtPlug(Plug):
	node : RenderGlobals = None
	pass
class CompositePlug(Plug):
	node : RenderGlobals = None
	pass
class CompositeThresholdPlug(Plug):
	node : RenderGlobals = None
	pass
class CreateIprFilePlug(Plug):
	node : RenderGlobals = None
	pass
class CurrentRendererPlug(Plug):
	node : RenderGlobals = None
	pass
class DefaultTraversalSetPlug(Plug):
	node : RenderGlobals = None
	pass
class EnableDefaultLightPlug(Plug):
	node : RenderGlobals = None
	pass
class EnableDepthMapsPlug(Plug):
	node : RenderGlobals = None
	pass
class EnableStrokeRenderPlug(Plug):
	node : RenderGlobals = None
	pass
class EndFramePlug(Plug):
	node : RenderGlobals = None
	pass
class EvenFieldExtPlug(Plug):
	node : RenderGlobals = None
	pass
class ExrCompressionPlug(Plug):
	node : RenderGlobals = None
	pass
class ExrPixelTypePlug(Plug):
	node : RenderGlobals = None
	pass
class ExtensionPaddingPlug(Plug):
	node : RenderGlobals = None
	pass
class FieldExtControlPlug(Plug):
	node : RenderGlobals = None
	pass
class FogGeometryPlug(Plug):
	node : RenderGlobals = None
	pass
class ForceTileSizePlug(Plug):
	node : RenderGlobals = None
	pass
class GammaCorrectionPlug(Plug):
	node : RenderGlobals = None
	pass
class GeometryVectorPlug(Plug):
	node : RenderGlobals = None
	pass
class HyperShadeBinListPlug(Plug):
	node : RenderGlobals = None
	pass
class IgnoreFilmGatePlug(Plug):
	node : RenderGlobals = None
	pass
class ImageFilePrefixPlug(Plug):
	node : RenderGlobals = None
	pass
class ImageFormatPlug(Plug):
	node : RenderGlobals = None
	pass
class ImfPluginKeyPlug(Plug):
	node : RenderGlobals = None
	pass
class InputColorProfilePlug(Plug):
	node : RenderGlobals = None
	pass
class InterruptFrequencyPlug(Plug):
	node : RenderGlobals = None
	pass
class IprRenderMotionBlurPlug(Plug):
	node : RenderGlobals = None
	pass
class IprRenderShadingPlug(Plug):
	node : RenderGlobals = None
	pass
class IprRenderShadowMapsPlug(Plug):
	node : RenderGlobals = None
	pass
class IprShadowPassPlug(Plug):
	node : RenderGlobals = None
	pass
class JitterFinalColorPlug(Plug):
	node : RenderGlobals = None
	pass
class KeepMotionVectorPlug(Plug):
	node : RenderGlobals = None
	pass
class LeafPrimitivesPlug(Plug):
	node : RenderGlobals = None
	pass
class LeftRegionPlug(Plug):
	node : RenderGlobals = None
	pass
class LogRenderPerformancePlug(Plug):
	node : RenderGlobals = None
	pass
class MacCodecPlug(Plug):
	node : RenderGlobals = None
	pass
class MacDepthPlug(Plug):
	node : RenderGlobals = None
	pass
class MacQualPlug(Plug):
	node : RenderGlobals = None
	pass
class MatteOpacityUsesTransparencyPlug(Plug):
	node : RenderGlobals = None
	pass
class MaximumMemoryPlug(Plug):
	node : RenderGlobals = None
	pass
class ModifyExtensionPlug(Plug):
	node : RenderGlobals = None
	pass
class MotionBlurPlug(Plug):
	node : RenderGlobals = None
	pass
class MotionBlurByFramePlug(Plug):
	node : RenderGlobals = None
	pass
class MotionBlurShutterClosePlug(Plug):
	node : RenderGlobals = None
	pass
class MotionBlurShutterOpenPlug(Plug):
	node : RenderGlobals = None
	pass
class MotionBlurTypePlug(Plug):
	node : RenderGlobals = None
	pass
class MotionBlurUseShutterPlug(Plug):
	node : RenderGlobals = None
	pass
class MultiCamNamingModePlug(Plug):
	node : RenderGlobals = None
	pass
class NumCpusToUsePlug(Plug):
	node : RenderGlobals = None
	pass
class OddFieldExtPlug(Plug):
	node : RenderGlobals = None
	pass
class OnlyRenderStrokesPlug(Plug):
	node : RenderGlobals = None
	pass
class OptimizeInstancesPlug(Plug):
	node : RenderGlobals = None
	pass
class OutFormatControlPlug(Plug):
	node : RenderGlobals = None
	pass
class OutFormatExtPlug(Plug):
	node : RenderGlobals = None
	pass
class OutputColorProfilePlug(Plug):
	node : RenderGlobals = None
	pass
class OversamplePaintEffectsPlug(Plug):
	node : RenderGlobals = None
	pass
class OversamplePfxPostFilterPlug(Plug):
	node : RenderGlobals = None
	pass
class PeriodInExtPlug(Plug):
	node : RenderGlobals = None
	pass
class PostFogBlurPlug(Plug):
	node : RenderGlobals = None
	pass
class PostFurRenderMelPlug(Plug):
	node : RenderGlobals = None
	pass
class PostMelPlug(Plug):
	node : RenderGlobals = None
	pass
class PostRenderLayerMelPlug(Plug):
	node : RenderGlobals = None
	pass
class PostRenderMelPlug(Plug):
	node : RenderGlobals = None
	pass
class PreFurRenderMelPlug(Plug):
	node : RenderGlobals = None
	pass
class PreMelPlug(Plug):
	node : RenderGlobals = None
	pass
class PreRenderLayerMelPlug(Plug):
	node : RenderGlobals = None
	pass
class PreRenderMelPlug(Plug):
	node : RenderGlobals = None
	pass
class PutFrameBeforeExtPlug(Plug):
	node : RenderGlobals = None
	pass
class QualityPlug(Plug):
	node : RenderGlobals = None
	pass
class RaysSeeBackgroundPlug(Plug):
	node : RenderGlobals = None
	pass
class RecursionDepthPlug(Plug):
	node : RenderGlobals = None
	pass
class RenderAllPlug(Plug):
	node : RenderGlobals = None
	pass
class RenderLayerEnablePlug(Plug):
	node : RenderGlobals = None
	pass
class RenderVersionPlug(Plug):
	node : RenderGlobals = None
	pass
class RendercallbackPlug(Plug):
	node : RenderGlobals = None
	pass
class RenderedOutputPlug(Plug):
	node : RenderGlobals = None
	pass
class RenderingColorProfilePlug(Plug):
	node : RenderGlobals = None
	pass
class ResolutionPlug(Plug):
	node : RenderGlobals = None
	pass
class ReuseTessellationsPlug(Plug):
	node : RenderGlobals = None
	pass
class RightRegionPlug(Plug):
	node : RenderGlobals = None
	pass
class ShadingVectorPlug(Plug):
	node : RenderGlobals = None
	pass
class ShadowPassPlug(Plug):
	node : RenderGlobals = None
	pass
class ShadowsObeyLightLinkingPlug(Plug):
	node : RenderGlobals = None
	pass
class ShadowsObeyShadowLinkingPlug(Plug):
	node : RenderGlobals = None
	pass
class SkipExistingFramesPlug(Plug):
	node : RenderGlobals = None
	pass
class SmoothColorPlug(Plug):
	node : RenderGlobals = None
	pass
class SmoothValuePlug(Plug):
	node : RenderGlobals = None
	pass
class StartExtensionPlug(Plug):
	node : RenderGlobals = None
	pass
class StartFramePlug(Plug):
	node : RenderGlobals = None
	pass
class StrokesDepthFilePlug(Plug):
	node : RenderGlobals = None
	pass
class SubdivisionHashSizePlug(Plug):
	node : RenderGlobals = None
	pass
class SubdivisionPowerPlug(Plug):
	node : RenderGlobals = None
	pass
class SwatchCameraPlug(Plug):
	node : RenderGlobals = None
	pass
class TiffCompressionPlug(Plug):
	node : RenderGlobals = None
	pass
class TileHeightPlug(Plug):
	node : RenderGlobals = None
	pass
class TileWidthPlug(Plug):
	node : RenderGlobals = None
	pass
class TopRegionPlug(Plug):
	node : RenderGlobals = None
	pass
class UseBlur2DMemoryCapPlug(Plug):
	node : RenderGlobals = None
	pass
class UseDisplacementBoundingBoxPlug(Plug):
	node : RenderGlobals = None
	pass
class UseFileCachePlug(Plug):
	node : RenderGlobals = None
	pass
class UseFrameExtPlug(Plug):
	node : RenderGlobals = None
	pass
class UseMayaFileNamePlug(Plug):
	node : RenderGlobals = None
	pass
class UseRenderRegionPlug(Plug):
	node : RenderGlobals = None
	pass
# endregion


# define node class
class RenderGlobals(_BASE_):
	animation_ : AnimationPlug = PlugDescriptor("animation")
	animationRange_ : AnimationRangePlug = PlugDescriptor("animationRange")
	applyFogInPost_ : ApplyFogInPostPlug = PlugDescriptor("applyFogInPost")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	bitDepth_ : BitDepthPlug = PlugDescriptor("bitDepth")
	blur2DMemoryCap_ : Blur2DMemoryCapPlug = PlugDescriptor("blur2DMemoryCap")
	blurLength_ : BlurLengthPlug = PlugDescriptor("blurLength")
	blurSharpness_ : BlurSharpnessPlug = PlugDescriptor("blurSharpness")
	bottomRegion_ : BottomRegionPlug = PlugDescriptor("bottomRegion")
	bufferName_ : BufferNamePlug = PlugDescriptor("bufferName")
	byExtension_ : ByExtensionPlug = PlugDescriptor("byExtension")
	byFrame_ : ByFramePlug = PlugDescriptor("byFrame")
	byFrameStep_ : ByFrameStepPlug = PlugDescriptor("byFrameStep")
	clipFinalShadedColor_ : ClipFinalShadedColorPlug = PlugDescriptor("clipFinalShadedColor")
	colorProfileEnabled_ : ColorProfileEnabledPlug = PlugDescriptor("colorProfileEnabled")
	comFrrt_ : ComFrrtPlug = PlugDescriptor("comFrrt")
	composite_ : CompositePlug = PlugDescriptor("composite")
	compositeThreshold_ : CompositeThresholdPlug = PlugDescriptor("compositeThreshold")
	createIprFile_ : CreateIprFilePlug = PlugDescriptor("createIprFile")
	currentRenderer_ : CurrentRendererPlug = PlugDescriptor("currentRenderer")
	defaultTraversalSet_ : DefaultTraversalSetPlug = PlugDescriptor("defaultTraversalSet")
	enableDefaultLight_ : EnableDefaultLightPlug = PlugDescriptor("enableDefaultLight")
	enableDepthMaps_ : EnableDepthMapsPlug = PlugDescriptor("enableDepthMaps")
	enableStrokeRender_ : EnableStrokeRenderPlug = PlugDescriptor("enableStrokeRender")
	endFrame_ : EndFramePlug = PlugDescriptor("endFrame")
	evenFieldExt_ : EvenFieldExtPlug = PlugDescriptor("evenFieldExt")
	exrCompression_ : ExrCompressionPlug = PlugDescriptor("exrCompression")
	exrPixelType_ : ExrPixelTypePlug = PlugDescriptor("exrPixelType")
	extensionPadding_ : ExtensionPaddingPlug = PlugDescriptor("extensionPadding")
	fieldExtControl_ : FieldExtControlPlug = PlugDescriptor("fieldExtControl")
	fogGeometry_ : FogGeometryPlug = PlugDescriptor("fogGeometry")
	forceTileSize_ : ForceTileSizePlug = PlugDescriptor("forceTileSize")
	gammaCorrection_ : GammaCorrectionPlug = PlugDescriptor("gammaCorrection")
	geometryVector_ : GeometryVectorPlug = PlugDescriptor("geometryVector")
	hyperShadeBinList_ : HyperShadeBinListPlug = PlugDescriptor("hyperShadeBinList")
	ignoreFilmGate_ : IgnoreFilmGatePlug = PlugDescriptor("ignoreFilmGate")
	imageFilePrefix_ : ImageFilePrefixPlug = PlugDescriptor("imageFilePrefix")
	imageFormat_ : ImageFormatPlug = PlugDescriptor("imageFormat")
	imfPluginKey_ : ImfPluginKeyPlug = PlugDescriptor("imfPluginKey")
	inputColorProfile_ : InputColorProfilePlug = PlugDescriptor("inputColorProfile")
	interruptFrequency_ : InterruptFrequencyPlug = PlugDescriptor("interruptFrequency")
	iprRenderMotionBlur_ : IprRenderMotionBlurPlug = PlugDescriptor("iprRenderMotionBlur")
	iprRenderShading_ : IprRenderShadingPlug = PlugDescriptor("iprRenderShading")
	iprRenderShadowMaps_ : IprRenderShadowMapsPlug = PlugDescriptor("iprRenderShadowMaps")
	iprShadowPass_ : IprShadowPassPlug = PlugDescriptor("iprShadowPass")
	jitterFinalColor_ : JitterFinalColorPlug = PlugDescriptor("jitterFinalColor")
	keepMotionVector_ : KeepMotionVectorPlug = PlugDescriptor("keepMotionVector")
	leafPrimitives_ : LeafPrimitivesPlug = PlugDescriptor("leafPrimitives")
	leftRegion_ : LeftRegionPlug = PlugDescriptor("leftRegion")
	logRenderPerformance_ : LogRenderPerformancePlug = PlugDescriptor("logRenderPerformance")
	macCodec_ : MacCodecPlug = PlugDescriptor("macCodec")
	macDepth_ : MacDepthPlug = PlugDescriptor("macDepth")
	macQual_ : MacQualPlug = PlugDescriptor("macQual")
	matteOpacityUsesTransparency_ : MatteOpacityUsesTransparencyPlug = PlugDescriptor("matteOpacityUsesTransparency")
	maximumMemory_ : MaximumMemoryPlug = PlugDescriptor("maximumMemory")
	modifyExtension_ : ModifyExtensionPlug = PlugDescriptor("modifyExtension")
	motionBlur_ : MotionBlurPlug = PlugDescriptor("motionBlur")
	motionBlurByFrame_ : MotionBlurByFramePlug = PlugDescriptor("motionBlurByFrame")
	motionBlurShutterClose_ : MotionBlurShutterClosePlug = PlugDescriptor("motionBlurShutterClose")
	motionBlurShutterOpen_ : MotionBlurShutterOpenPlug = PlugDescriptor("motionBlurShutterOpen")
	motionBlurType_ : MotionBlurTypePlug = PlugDescriptor("motionBlurType")
	motionBlurUseShutter_ : MotionBlurUseShutterPlug = PlugDescriptor("motionBlurUseShutter")
	multiCamNamingMode_ : MultiCamNamingModePlug = PlugDescriptor("multiCamNamingMode")
	numCpusToUse_ : NumCpusToUsePlug = PlugDescriptor("numCpusToUse")
	oddFieldExt_ : OddFieldExtPlug = PlugDescriptor("oddFieldExt")
	onlyRenderStrokes_ : OnlyRenderStrokesPlug = PlugDescriptor("onlyRenderStrokes")
	optimizeInstances_ : OptimizeInstancesPlug = PlugDescriptor("optimizeInstances")
	outFormatControl_ : OutFormatControlPlug = PlugDescriptor("outFormatControl")
	outFormatExt_ : OutFormatExtPlug = PlugDescriptor("outFormatExt")
	outputColorProfile_ : OutputColorProfilePlug = PlugDescriptor("outputColorProfile")
	oversamplePaintEffects_ : OversamplePaintEffectsPlug = PlugDescriptor("oversamplePaintEffects")
	oversamplePfxPostFilter_ : OversamplePfxPostFilterPlug = PlugDescriptor("oversamplePfxPostFilter")
	periodInExt_ : PeriodInExtPlug = PlugDescriptor("periodInExt")
	postFogBlur_ : PostFogBlurPlug = PlugDescriptor("postFogBlur")
	postFurRenderMel_ : PostFurRenderMelPlug = PlugDescriptor("postFurRenderMel")
	postMel_ : PostMelPlug = PlugDescriptor("postMel")
	postRenderLayerMel_ : PostRenderLayerMelPlug = PlugDescriptor("postRenderLayerMel")
	postRenderMel_ : PostRenderMelPlug = PlugDescriptor("postRenderMel")
	preFurRenderMel_ : PreFurRenderMelPlug = PlugDescriptor("preFurRenderMel")
	preMel_ : PreMelPlug = PlugDescriptor("preMel")
	preRenderLayerMel_ : PreRenderLayerMelPlug = PlugDescriptor("preRenderLayerMel")
	preRenderMel_ : PreRenderMelPlug = PlugDescriptor("preRenderMel")
	putFrameBeforeExt_ : PutFrameBeforeExtPlug = PlugDescriptor("putFrameBeforeExt")
	quality_ : QualityPlug = PlugDescriptor("quality")
	raysSeeBackground_ : RaysSeeBackgroundPlug = PlugDescriptor("raysSeeBackground")
	recursionDepth_ : RecursionDepthPlug = PlugDescriptor("recursionDepth")
	renderAll_ : RenderAllPlug = PlugDescriptor("renderAll")
	renderLayerEnable_ : RenderLayerEnablePlug = PlugDescriptor("renderLayerEnable")
	renderVersion_ : RenderVersionPlug = PlugDescriptor("renderVersion")
	rendercallback_ : RendercallbackPlug = PlugDescriptor("rendercallback")
	renderedOutput_ : RenderedOutputPlug = PlugDescriptor("renderedOutput")
	renderingColorProfile_ : RenderingColorProfilePlug = PlugDescriptor("renderingColorProfile")
	resolution_ : ResolutionPlug = PlugDescriptor("resolution")
	reuseTessellations_ : ReuseTessellationsPlug = PlugDescriptor("reuseTessellations")
	rightRegion_ : RightRegionPlug = PlugDescriptor("rightRegion")
	shadingVector_ : ShadingVectorPlug = PlugDescriptor("shadingVector")
	shadowPass_ : ShadowPassPlug = PlugDescriptor("shadowPass")
	shadowsObeyLightLinking_ : ShadowsObeyLightLinkingPlug = PlugDescriptor("shadowsObeyLightLinking")
	shadowsObeyShadowLinking_ : ShadowsObeyShadowLinkingPlug = PlugDescriptor("shadowsObeyShadowLinking")
	skipExistingFrames_ : SkipExistingFramesPlug = PlugDescriptor("skipExistingFrames")
	smoothColor_ : SmoothColorPlug = PlugDescriptor("smoothColor")
	smoothValue_ : SmoothValuePlug = PlugDescriptor("smoothValue")
	startExtension_ : StartExtensionPlug = PlugDescriptor("startExtension")
	startFrame_ : StartFramePlug = PlugDescriptor("startFrame")
	strokesDepthFile_ : StrokesDepthFilePlug = PlugDescriptor("strokesDepthFile")
	subdivisionHashSize_ : SubdivisionHashSizePlug = PlugDescriptor("subdivisionHashSize")
	subdivisionPower_ : SubdivisionPowerPlug = PlugDescriptor("subdivisionPower")
	swatchCamera_ : SwatchCameraPlug = PlugDescriptor("swatchCamera")
	tiffCompression_ : TiffCompressionPlug = PlugDescriptor("tiffCompression")
	tileHeight_ : TileHeightPlug = PlugDescriptor("tileHeight")
	tileWidth_ : TileWidthPlug = PlugDescriptor("tileWidth")
	topRegion_ : TopRegionPlug = PlugDescriptor("topRegion")
	useBlur2DMemoryCap_ : UseBlur2DMemoryCapPlug = PlugDescriptor("useBlur2DMemoryCap")
	useDisplacementBoundingBox_ : UseDisplacementBoundingBoxPlug = PlugDescriptor("useDisplacementBoundingBox")
	useFileCache_ : UseFileCachePlug = PlugDescriptor("useFileCache")
	useFrameExt_ : UseFrameExtPlug = PlugDescriptor("useFrameExt")
	useMayaFileName_ : UseMayaFileNamePlug = PlugDescriptor("useMayaFileName")
	useRenderRegion_ : UseRenderRegionPlug = PlugDescriptor("useRenderRegion")

	# node attributes

	typeName = "renderGlobals"
	apiTypeInt = 523
	apiTypeStr = "kRenderGlobals"
	typeIdInt = 1380404290
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["animation", "animationRange", "applyFogInPost", "binMembership", "bitDepth", "blur2DMemoryCap", "blurLength", "blurSharpness", "bottomRegion", "bufferName", "byExtension", "byFrame", "byFrameStep", "clipFinalShadedColor", "colorProfileEnabled", "comFrrt", "composite", "compositeThreshold", "createIprFile", "currentRenderer", "defaultTraversalSet", "enableDefaultLight", "enableDepthMaps", "enableStrokeRender", "endFrame", "evenFieldExt", "exrCompression", "exrPixelType", "extensionPadding", "fieldExtControl", "fogGeometry", "forceTileSize", "gammaCorrection", "geometryVector", "hyperShadeBinList", "ignoreFilmGate", "imageFilePrefix", "imageFormat", "imfPluginKey", "inputColorProfile", "interruptFrequency", "iprRenderMotionBlur", "iprRenderShading", "iprRenderShadowMaps", "iprShadowPass", "jitterFinalColor", "keepMotionVector", "leafPrimitives", "leftRegion", "logRenderPerformance", "macCodec", "macDepth", "macQual", "matteOpacityUsesTransparency", "maximumMemory", "modifyExtension", "motionBlur", "motionBlurByFrame", "motionBlurShutterClose", "motionBlurShutterOpen", "motionBlurType", "motionBlurUseShutter", "multiCamNamingMode", "numCpusToUse", "oddFieldExt", "onlyRenderStrokes", "optimizeInstances", "outFormatControl", "outFormatExt", "outputColorProfile", "oversamplePaintEffects", "oversamplePfxPostFilter", "periodInExt", "postFogBlur", "postFurRenderMel", "postMel", "postRenderLayerMel", "postRenderMel", "preFurRenderMel", "preMel", "preRenderLayerMel", "preRenderMel", "putFrameBeforeExt", "quality", "raysSeeBackground", "recursionDepth", "renderAll", "renderLayerEnable", "renderVersion", "rendercallback", "renderedOutput", "renderingColorProfile", "resolution", "reuseTessellations", "rightRegion", "shadingVector", "shadowPass", "shadowsObeyLightLinking", "shadowsObeyShadowLinking", "skipExistingFrames", "smoothColor", "smoothValue", "startExtension", "startFrame", "strokesDepthFile", "subdivisionHashSize", "subdivisionPower", "swatchCamera", "tiffCompression", "tileHeight", "tileWidth", "topRegion", "useBlur2DMemoryCap", "useDisplacementBoundingBox", "useFileCache", "useFrameExt", "useMayaFileName", "useRenderRegion"]
	nodeLeafPlugs = ["animation", "animationRange", "applyFogInPost", "binMembership", "bitDepth", "blur2DMemoryCap", "blurLength", "blurSharpness", "bottomRegion", "bufferName", "byExtension", "byFrame", "byFrameStep", "clipFinalShadedColor", "colorProfileEnabled", "comFrrt", "composite", "compositeThreshold", "createIprFile", "currentRenderer", "defaultTraversalSet", "enableDefaultLight", "enableDepthMaps", "enableStrokeRender", "endFrame", "evenFieldExt", "exrCompression", "exrPixelType", "extensionPadding", "fieldExtControl", "fogGeometry", "forceTileSize", "gammaCorrection", "geometryVector", "hyperShadeBinList", "ignoreFilmGate", "imageFilePrefix", "imageFormat", "imfPluginKey", "inputColorProfile", "interruptFrequency", "iprRenderMotionBlur", "iprRenderShading", "iprRenderShadowMaps", "iprShadowPass", "jitterFinalColor", "keepMotionVector", "leafPrimitives", "leftRegion", "logRenderPerformance", "macCodec", "macDepth", "macQual", "matteOpacityUsesTransparency", "maximumMemory", "modifyExtension", "motionBlur", "motionBlurByFrame", "motionBlurShutterClose", "motionBlurShutterOpen", "motionBlurType", "motionBlurUseShutter", "multiCamNamingMode", "numCpusToUse", "oddFieldExt", "onlyRenderStrokes", "optimizeInstances", "outFormatControl", "outFormatExt", "outputColorProfile", "oversamplePaintEffects", "oversamplePfxPostFilter", "periodInExt", "postFogBlur", "postFurRenderMel", "postMel", "postRenderLayerMel", "postRenderMel", "preFurRenderMel", "preMel", "preRenderLayerMel", "preRenderMel", "putFrameBeforeExt", "quality", "raysSeeBackground", "recursionDepth", "renderAll", "renderLayerEnable", "renderVersion", "rendercallback", "renderedOutput", "renderingColorProfile", "resolution", "reuseTessellations", "rightRegion", "shadingVector", "shadowPass", "shadowsObeyLightLinking", "shadowsObeyShadowLinking", "skipExistingFrames", "smoothColor", "smoothValue", "startExtension", "startFrame", "strokesDepthFile", "subdivisionHashSize", "subdivisionPower", "swatchCamera", "tiffCompression", "tileHeight", "tileWidth", "topRegion", "useBlur2DMemoryCap", "useDisplacementBoundingBox", "useFileCache", "useFrameExt", "useMayaFileName", "useRenderRegion"]
	pass

