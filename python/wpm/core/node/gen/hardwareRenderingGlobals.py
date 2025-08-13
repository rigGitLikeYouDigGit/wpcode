

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
_BASE_ = retriever.getNodeCls("_BASE_")
assert _BASE_
if T.TYPE_CHECKING:
	from .. import _BASE_

# add node doc



# region plug type defs
class AlphaCutPrepassPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class LightingModePlug(Plug):
	parent : BatchRenderControlsPlug = PlugDescriptor("batchRenderControls")
	node : HardwareRenderingGlobals = None
	pass
class ObjectTypeFilterNameArrayPlug(Plug):
	parent : BatchRenderControlsPlug = PlugDescriptor("batchRenderControls")
	node : HardwareRenderingGlobals = None
	pass
class ObjectTypeFilterValueArrayPlug(Plug):
	parent : BatchRenderControlsPlug = PlugDescriptor("batchRenderControls")
	node : HardwareRenderingGlobals = None
	pass
class PluginObjectTypeFilterNameArrayPlug(Plug):
	parent : BatchRenderControlsPlug = PlugDescriptor("batchRenderControls")
	node : HardwareRenderingGlobals = None
	pass
class PluginObjectTypeFilterValueArrayPlug(Plug):
	parent : BatchRenderControlsPlug = PlugDescriptor("batchRenderControls")
	node : HardwareRenderingGlobals = None
	pass
class RenderModePlug(Plug):
	parent : BatchRenderControlsPlug = PlugDescriptor("batchRenderControls")
	node : HardwareRenderingGlobals = None
	pass
class BatchRenderControlsPlug(Plug):
	lightingMode_ : LightingModePlug = PlugDescriptor("lightingMode")
	lm_ : LightingModePlug = PlugDescriptor("lightingMode")
	objectTypeFilterNameArray_ : ObjectTypeFilterNameArrayPlug = PlugDescriptor("objectTypeFilterNameArray")
	otfna_ : ObjectTypeFilterNameArrayPlug = PlugDescriptor("objectTypeFilterNameArray")
	objectTypeFilterValueArray_ : ObjectTypeFilterValueArrayPlug = PlugDescriptor("objectTypeFilterValueArray")
	otfva_ : ObjectTypeFilterValueArrayPlug = PlugDescriptor("objectTypeFilterValueArray")
	pluginObjectTypeFilterNameArray_ : PluginObjectTypeFilterNameArrayPlug = PlugDescriptor("pluginObjectTypeFilterNameArray")
	potfna_ : PluginObjectTypeFilterNameArrayPlug = PlugDescriptor("pluginObjectTypeFilterNameArray")
	pluginObjectTypeFilterValueArray_ : PluginObjectTypeFilterValueArrayPlug = PlugDescriptor("pluginObjectTypeFilterValueArray")
	potfva_ : PluginObjectTypeFilterValueArrayPlug = PlugDescriptor("pluginObjectTypeFilterValueArray")
	renderMode_ : RenderModePlug = PlugDescriptor("renderMode")
	rm_ : RenderModePlug = PlugDescriptor("renderMode")
	node : HardwareRenderingGlobals = None
	pass
class BinMembershipPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class BloomAmountPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class BloomEnablePlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class BloomFilterPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class BloomFilterAuxPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class BloomFilterRadiusPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class BloomThresholdPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class BumpBakeResolutionPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class ColorBakeResolutionPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class CompressSharedVertexDataPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class ConsolidateWorldPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class CurrentRendererNamePlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class CustomUVBorderColorBPlug(Plug):
	parent : CustomUVBorderColorPlug = PlugDescriptor("customUVBorderColor")
	node : HardwareRenderingGlobals = None
	pass
class CustomUVBorderColorGPlug(Plug):
	parent : CustomUVBorderColorPlug = PlugDescriptor("customUVBorderColor")
	node : HardwareRenderingGlobals = None
	pass
class CustomUVBorderColorRPlug(Plug):
	parent : CustomUVBorderColorPlug = PlugDescriptor("customUVBorderColor")
	node : HardwareRenderingGlobals = None
	pass
class CustomUVBorderColorPlug(Plug):
	customUVBorderColorB_ : CustomUVBorderColorBPlug = PlugDescriptor("customUVBorderColorB")
	uvbcb_ : CustomUVBorderColorBPlug = PlugDescriptor("customUVBorderColorB")
	customUVBorderColorG_ : CustomUVBorderColorGPlug = PlugDescriptor("customUVBorderColorG")
	uvbcg_ : CustomUVBorderColorGPlug = PlugDescriptor("customUVBorderColorG")
	customUVBorderColorR_ : CustomUVBorderColorRPlug = PlugDescriptor("customUVBorderColorR")
	uvbcr_ : CustomUVBorderColorRPlug = PlugDescriptor("customUVBorderColorR")
	node : HardwareRenderingGlobals = None
	pass
class EnableTextureMaxResPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class FloatingPointRTEnablePlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class FloatingPointRTFormatPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class GammaCorrectionEnablePlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class GammaValuePlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class HoldOutDetailModePlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class HoldOutModePlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class HwFogAlphaPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class HwFogColorBPlug(Plug):
	parent : HwFogColorPlug = PlugDescriptor("hwFogColor")
	node : HardwareRenderingGlobals = None
	pass
class HwFogColorGPlug(Plug):
	parent : HwFogColorPlug = PlugDescriptor("hwFogColor")
	node : HardwareRenderingGlobals = None
	pass
class HwFogColorRPlug(Plug):
	parent : HwFogColorPlug = PlugDescriptor("hwFogColor")
	node : HardwareRenderingGlobals = None
	pass
class HwFogColorPlug(Plug):
	hwFogColorB_ : HwFogColorBPlug = PlugDescriptor("hwFogColorB")
	hfcb_ : HwFogColorBPlug = PlugDescriptor("hwFogColorB")
	hwFogColorG_ : HwFogColorGPlug = PlugDescriptor("hwFogColorG")
	hfcg_ : HwFogColorGPlug = PlugDescriptor("hwFogColorG")
	hwFogColorR_ : HwFogColorRPlug = PlugDescriptor("hwFogColorR")
	hfcr_ : HwFogColorRPlug = PlugDescriptor("hwFogColorR")
	node : HardwareRenderingGlobals = None
	pass
class HwFogDensityPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class HwFogEnablePlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class HwFogEndPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class HwFogFalloffPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class HwFogStartPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class HwInstancingPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class IsCustomUVBorderColorPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class LineAAEnablePlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class MaxHardwareLightsPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurAtlasSizeXPlug(Plug):
	parent : MotionBlurAtlasSizePlug = PlugDescriptor("motionBlurAtlasSize")
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurAtlasSizeYPlug(Plug):
	parent : MotionBlurAtlasSizePlug = PlugDescriptor("motionBlurAtlasSize")
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurAtlasSizePlug(Plug):
	motionBlurAtlasSizeX_ : MotionBlurAtlasSizeXPlug = PlugDescriptor("motionBlurAtlasSizeX")
	mbasx_ : MotionBlurAtlasSizeXPlug = PlugDescriptor("motionBlurAtlasSizeX")
	motionBlurAtlasSizeY_ : MotionBlurAtlasSizeYPlug = PlugDescriptor("motionBlurAtlasSizeY")
	mbasy_ : MotionBlurAtlasSizeYPlug = PlugDescriptor("motionBlurAtlasSizeY")
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurCurvedPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurEnablePlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurFadeAmountPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurFadeEmphasisPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurFadeFilterPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurFadeTintBPlug(Plug):
	parent : MotionBlurFadeTintPlug = PlugDescriptor("motionBlurFadeTint")
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurFadeTintGPlug(Plug):
	parent : MotionBlurFadeTintPlug = PlugDescriptor("motionBlurFadeTint")
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurFadeTintRPlug(Plug):
	parent : MotionBlurFadeTintPlug = PlugDescriptor("motionBlurFadeTint")
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurFadeTintPlug(Plug):
	motionBlurFadeTintB_ : MotionBlurFadeTintBPlug = PlugDescriptor("motionBlurFadeTintB")
	mbftb_ : MotionBlurFadeTintBPlug = PlugDescriptor("motionBlurFadeTintB")
	motionBlurFadeTintG_ : MotionBlurFadeTintGPlug = PlugDescriptor("motionBlurFadeTintG")
	mbftg_ : MotionBlurFadeTintGPlug = PlugDescriptor("motionBlurFadeTintG")
	motionBlurFadeTintR_ : MotionBlurFadeTintRPlug = PlugDescriptor("motionBlurFadeTintR")
	mbftr_ : MotionBlurFadeTintRPlug = PlugDescriptor("motionBlurFadeTintR")
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurFadeTintAPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurMultiframeChartSizeXPlug(Plug):
	parent : MotionBlurMultiframeChartSizePlug = PlugDescriptor("motionBlurMultiframeChartSize")
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurMultiframeChartSizeYPlug(Plug):
	parent : MotionBlurMultiframeChartSizePlug = PlugDescriptor("motionBlurMultiframeChartSize")
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurMultiframeChartSizePlug(Plug):
	motionBlurMultiframeChartSizeX_ : MotionBlurMultiframeChartSizeXPlug = PlugDescriptor("motionBlurMultiframeChartSizeX")
	mbcsx_ : MotionBlurMultiframeChartSizeXPlug = PlugDescriptor("motionBlurMultiframeChartSizeX")
	motionBlurMultiframeChartSizeY_ : MotionBlurMultiframeChartSizeYPlug = PlugDescriptor("motionBlurMultiframeChartSizeY")
	mbcsy_ : MotionBlurMultiframeChartSizeYPlug = PlugDescriptor("motionBlurMultiframeChartSizeY")
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurMultiframeEnablePlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurSampleCountPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurShutterOpenFractionPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class MotionBlurTypePlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class MultiSampleCountPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class MultiSampleEnablePlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class MultiSampleQualityPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class QuadDrawAlwaysOnTopPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class QuadDrawOverrideColorBPlug(Plug):
	parent : QuadDrawOverrideColorPlug = PlugDescriptor("quadDrawOverrideColor")
	node : HardwareRenderingGlobals = None
	pass
class QuadDrawOverrideColorGPlug(Plug):
	parent : QuadDrawOverrideColorPlug = PlugDescriptor("quadDrawOverrideColor")
	node : HardwareRenderingGlobals = None
	pass
class QuadDrawOverrideColorRPlug(Plug):
	parent : QuadDrawOverrideColorPlug = PlugDescriptor("quadDrawOverrideColor")
	node : HardwareRenderingGlobals = None
	pass
class QuadDrawOverrideColorPlug(Plug):
	quadDrawOverrideColorB_ : QuadDrawOverrideColorBPlug = PlugDescriptor("quadDrawOverrideColorB")
	qdocb_ : QuadDrawOverrideColorBPlug = PlugDescriptor("quadDrawOverrideColorB")
	quadDrawOverrideColorG_ : QuadDrawOverrideColorGPlug = PlugDescriptor("quadDrawOverrideColorG")
	qdocg_ : QuadDrawOverrideColorGPlug = PlugDescriptor("quadDrawOverrideColorG")
	quadDrawOverrideColorR_ : QuadDrawOverrideColorRPlug = PlugDescriptor("quadDrawOverrideColorR")
	qdocr_ : QuadDrawOverrideColorRPlug = PlugDescriptor("quadDrawOverrideColorR")
	node : HardwareRenderingGlobals = None
	pass
class QuadDrawOverrideTransparencyPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class RenderDepthOfFieldPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class RenderOverrideNamePlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class SingleSidedLightingPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class SsaoAmountPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class SsaoEnablePlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class SsaoFilterPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class SsaoFilterRadiusPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class SsaoRadiusPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class SsaoSamplesPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class TextureAutoMaxResolutionPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class TextureMaxResModePlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class TextureMaxResolutionPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class TransparencyAlgorithmPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class TransparencyQualityPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class TransparentShadowPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class UseMaximumHardwareLightsPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class VertexAnimationCachePlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class XrayJointDisplayPlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
class XrayModePlug(Plug):
	node : HardwareRenderingGlobals = None
	pass
# endregion


# define node class
class HardwareRenderingGlobals(_BASE_):
	alphaCutPrepass_ : AlphaCutPrepassPlug = PlugDescriptor("alphaCutPrepass")
	lightingMode_ : LightingModePlug = PlugDescriptor("lightingMode")
	objectTypeFilterNameArray_ : ObjectTypeFilterNameArrayPlug = PlugDescriptor("objectTypeFilterNameArray")
	objectTypeFilterValueArray_ : ObjectTypeFilterValueArrayPlug = PlugDescriptor("objectTypeFilterValueArray")
	pluginObjectTypeFilterNameArray_ : PluginObjectTypeFilterNameArrayPlug = PlugDescriptor("pluginObjectTypeFilterNameArray")
	pluginObjectTypeFilterValueArray_ : PluginObjectTypeFilterValueArrayPlug = PlugDescriptor("pluginObjectTypeFilterValueArray")
	renderMode_ : RenderModePlug = PlugDescriptor("renderMode")
	batchRenderControls_ : BatchRenderControlsPlug = PlugDescriptor("batchRenderControls")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	bloomAmount_ : BloomAmountPlug = PlugDescriptor("bloomAmount")
	bloomEnable_ : BloomEnablePlug = PlugDescriptor("bloomEnable")
	bloomFilter_ : BloomFilterPlug = PlugDescriptor("bloomFilter")
	bloomFilterAux_ : BloomFilterAuxPlug = PlugDescriptor("bloomFilterAux")
	bloomFilterRadius_ : BloomFilterRadiusPlug = PlugDescriptor("bloomFilterRadius")
	bloomThreshold_ : BloomThresholdPlug = PlugDescriptor("bloomThreshold")
	bumpBakeResolution_ : BumpBakeResolutionPlug = PlugDescriptor("bumpBakeResolution")
	colorBakeResolution_ : ColorBakeResolutionPlug = PlugDescriptor("colorBakeResolution")
	compressSharedVertexData_ : CompressSharedVertexDataPlug = PlugDescriptor("compressSharedVertexData")
	consolidateWorld_ : ConsolidateWorldPlug = PlugDescriptor("consolidateWorld")
	currentRendererName_ : CurrentRendererNamePlug = PlugDescriptor("currentRendererName")
	customUVBorderColorB_ : CustomUVBorderColorBPlug = PlugDescriptor("customUVBorderColorB")
	customUVBorderColorG_ : CustomUVBorderColorGPlug = PlugDescriptor("customUVBorderColorG")
	customUVBorderColorR_ : CustomUVBorderColorRPlug = PlugDescriptor("customUVBorderColorR")
	customUVBorderColor_ : CustomUVBorderColorPlug = PlugDescriptor("customUVBorderColor")
	enableTextureMaxRes_ : EnableTextureMaxResPlug = PlugDescriptor("enableTextureMaxRes")
	floatingPointRTEnable_ : FloatingPointRTEnablePlug = PlugDescriptor("floatingPointRTEnable")
	floatingPointRTFormat_ : FloatingPointRTFormatPlug = PlugDescriptor("floatingPointRTFormat")
	gammaCorrectionEnable_ : GammaCorrectionEnablePlug = PlugDescriptor("gammaCorrectionEnable")
	gammaValue_ : GammaValuePlug = PlugDescriptor("gammaValue")
	holdOutDetailMode_ : HoldOutDetailModePlug = PlugDescriptor("holdOutDetailMode")
	holdOutMode_ : HoldOutModePlug = PlugDescriptor("holdOutMode")
	hwFogAlpha_ : HwFogAlphaPlug = PlugDescriptor("hwFogAlpha")
	hwFogColorB_ : HwFogColorBPlug = PlugDescriptor("hwFogColorB")
	hwFogColorG_ : HwFogColorGPlug = PlugDescriptor("hwFogColorG")
	hwFogColorR_ : HwFogColorRPlug = PlugDescriptor("hwFogColorR")
	hwFogColor_ : HwFogColorPlug = PlugDescriptor("hwFogColor")
	hwFogDensity_ : HwFogDensityPlug = PlugDescriptor("hwFogDensity")
	hwFogEnable_ : HwFogEnablePlug = PlugDescriptor("hwFogEnable")
	hwFogEnd_ : HwFogEndPlug = PlugDescriptor("hwFogEnd")
	hwFogFalloff_ : HwFogFalloffPlug = PlugDescriptor("hwFogFalloff")
	hwFogStart_ : HwFogStartPlug = PlugDescriptor("hwFogStart")
	hwInstancing_ : HwInstancingPlug = PlugDescriptor("hwInstancing")
	isCustomUVBorderColor_ : IsCustomUVBorderColorPlug = PlugDescriptor("isCustomUVBorderColor")
	lineAAEnable_ : LineAAEnablePlug = PlugDescriptor("lineAAEnable")
	maxHardwareLights_ : MaxHardwareLightsPlug = PlugDescriptor("maxHardwareLights")
	motionBlurAtlasSizeX_ : MotionBlurAtlasSizeXPlug = PlugDescriptor("motionBlurAtlasSizeX")
	motionBlurAtlasSizeY_ : MotionBlurAtlasSizeYPlug = PlugDescriptor("motionBlurAtlasSizeY")
	motionBlurAtlasSize_ : MotionBlurAtlasSizePlug = PlugDescriptor("motionBlurAtlasSize")
	motionBlurCurved_ : MotionBlurCurvedPlug = PlugDescriptor("motionBlurCurved")
	motionBlurEnable_ : MotionBlurEnablePlug = PlugDescriptor("motionBlurEnable")
	motionBlurFadeAmount_ : MotionBlurFadeAmountPlug = PlugDescriptor("motionBlurFadeAmount")
	motionBlurFadeEmphasis_ : MotionBlurFadeEmphasisPlug = PlugDescriptor("motionBlurFadeEmphasis")
	motionBlurFadeFilter_ : MotionBlurFadeFilterPlug = PlugDescriptor("motionBlurFadeFilter")
	motionBlurFadeTintB_ : MotionBlurFadeTintBPlug = PlugDescriptor("motionBlurFadeTintB")
	motionBlurFadeTintG_ : MotionBlurFadeTintGPlug = PlugDescriptor("motionBlurFadeTintG")
	motionBlurFadeTintR_ : MotionBlurFadeTintRPlug = PlugDescriptor("motionBlurFadeTintR")
	motionBlurFadeTint_ : MotionBlurFadeTintPlug = PlugDescriptor("motionBlurFadeTint")
	motionBlurFadeTintA_ : MotionBlurFadeTintAPlug = PlugDescriptor("motionBlurFadeTintA")
	motionBlurMultiframeChartSizeX_ : MotionBlurMultiframeChartSizeXPlug = PlugDescriptor("motionBlurMultiframeChartSizeX")
	motionBlurMultiframeChartSizeY_ : MotionBlurMultiframeChartSizeYPlug = PlugDescriptor("motionBlurMultiframeChartSizeY")
	motionBlurMultiframeChartSize_ : MotionBlurMultiframeChartSizePlug = PlugDescriptor("motionBlurMultiframeChartSize")
	motionBlurMultiframeEnable_ : MotionBlurMultiframeEnablePlug = PlugDescriptor("motionBlurMultiframeEnable")
	motionBlurSampleCount_ : MotionBlurSampleCountPlug = PlugDescriptor("motionBlurSampleCount")
	motionBlurShutterOpenFraction_ : MotionBlurShutterOpenFractionPlug = PlugDescriptor("motionBlurShutterOpenFraction")
	motionBlurType_ : MotionBlurTypePlug = PlugDescriptor("motionBlurType")
	multiSampleCount_ : MultiSampleCountPlug = PlugDescriptor("multiSampleCount")
	multiSampleEnable_ : MultiSampleEnablePlug = PlugDescriptor("multiSampleEnable")
	multiSampleQuality_ : MultiSampleQualityPlug = PlugDescriptor("multiSampleQuality")
	quadDrawAlwaysOnTop_ : QuadDrawAlwaysOnTopPlug = PlugDescriptor("quadDrawAlwaysOnTop")
	quadDrawOverrideColorB_ : QuadDrawOverrideColorBPlug = PlugDescriptor("quadDrawOverrideColorB")
	quadDrawOverrideColorG_ : QuadDrawOverrideColorGPlug = PlugDescriptor("quadDrawOverrideColorG")
	quadDrawOverrideColorR_ : QuadDrawOverrideColorRPlug = PlugDescriptor("quadDrawOverrideColorR")
	quadDrawOverrideColor_ : QuadDrawOverrideColorPlug = PlugDescriptor("quadDrawOverrideColor")
	quadDrawOverrideTransparency_ : QuadDrawOverrideTransparencyPlug = PlugDescriptor("quadDrawOverrideTransparency")
	renderDepthOfField_ : RenderDepthOfFieldPlug = PlugDescriptor("renderDepthOfField")
	renderOverrideName_ : RenderOverrideNamePlug = PlugDescriptor("renderOverrideName")
	singleSidedLighting_ : SingleSidedLightingPlug = PlugDescriptor("singleSidedLighting")
	ssaoAmount_ : SsaoAmountPlug = PlugDescriptor("ssaoAmount")
	ssaoEnable_ : SsaoEnablePlug = PlugDescriptor("ssaoEnable")
	ssaoFilter_ : SsaoFilterPlug = PlugDescriptor("ssaoFilter")
	ssaoFilterRadius_ : SsaoFilterRadiusPlug = PlugDescriptor("ssaoFilterRadius")
	ssaoRadius_ : SsaoRadiusPlug = PlugDescriptor("ssaoRadius")
	ssaoSamples_ : SsaoSamplesPlug = PlugDescriptor("ssaoSamples")
	textureAutoMaxResolution_ : TextureAutoMaxResolutionPlug = PlugDescriptor("textureAutoMaxResolution")
	textureMaxResMode_ : TextureMaxResModePlug = PlugDescriptor("textureMaxResMode")
	textureMaxResolution_ : TextureMaxResolutionPlug = PlugDescriptor("textureMaxResolution")
	transparencyAlgorithm_ : TransparencyAlgorithmPlug = PlugDescriptor("transparencyAlgorithm")
	transparencyQuality_ : TransparencyQualityPlug = PlugDescriptor("transparencyQuality")
	transparentShadow_ : TransparentShadowPlug = PlugDescriptor("transparentShadow")
	useMaximumHardwareLights_ : UseMaximumHardwareLightsPlug = PlugDescriptor("useMaximumHardwareLights")
	vertexAnimationCache_ : VertexAnimationCachePlug = PlugDescriptor("vertexAnimationCache")
	xrayJointDisplay_ : XrayJointDisplayPlug = PlugDescriptor("xrayJointDisplay")
	xrayMode_ : XrayModePlug = PlugDescriptor("xrayMode")

	# node attributes

	typeName = "hardwareRenderingGlobals"
	apiTypeInt = 1071
	apiTypeStr = "kHardwareRenderingGlobals"
	typeIdInt = 1213354567
	MFnCls = om.MFnDependencyNode
	pass

