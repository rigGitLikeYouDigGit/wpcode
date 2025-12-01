

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
class BinMembershipPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class BlendSpecularWithAlphaPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class BumpTextureResolutionPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class ColorTextureResolutionPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class CullingPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class CullingThresholdPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class EnableAcceleratedMultiSamplingPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class EnableEdgeAntiAliasingPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class EnableGeometryMaskPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class EnableHighQualityLightingPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class EnableMotionBlurPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class EnableNonPowerOfTwoTexturePlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class FrameBufferFormatPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class GraphicsHardwareGeometryCachingDataPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class GraphicsHardwareGeometryCachingIndexingPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class HardwareCodecPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class HardwareDepthPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class HardwareEnvironmentLookupPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class HardwareFrameRatePlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class HardwareQualPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class LightIntensityThresholdPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class MaximumGeometryCacheSizePlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class MotionBlurByFramePlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class NumberOfExposuresPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class NumberOfSamplesPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class ShadingModelPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class ShadowsObeyLightLinkingPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class ShadowsObeyShadowLinkingPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class SmallObjectCullingPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class TextureCompressionPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class TransparencySortingPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class TransparentShadowCastingPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class WriteAlphaAsColorPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
class WriteZDepthAsColorPlug(Plug):
	node : HardwareRenderGlobals = None
	pass
# endregion


# define node class
class HardwareRenderGlobals(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	blendSpecularWithAlpha_ : BlendSpecularWithAlphaPlug = PlugDescriptor("blendSpecularWithAlpha")
	bumpTextureResolution_ : BumpTextureResolutionPlug = PlugDescriptor("bumpTextureResolution")
	colorTextureResolution_ : ColorTextureResolutionPlug = PlugDescriptor("colorTextureResolution")
	culling_ : CullingPlug = PlugDescriptor("culling")
	cullingThreshold_ : CullingThresholdPlug = PlugDescriptor("cullingThreshold")
	enableAcceleratedMultiSampling_ : EnableAcceleratedMultiSamplingPlug = PlugDescriptor("enableAcceleratedMultiSampling")
	enableEdgeAntiAliasing_ : EnableEdgeAntiAliasingPlug = PlugDescriptor("enableEdgeAntiAliasing")
	enableGeometryMask_ : EnableGeometryMaskPlug = PlugDescriptor("enableGeometryMask")
	enableHighQualityLighting_ : EnableHighQualityLightingPlug = PlugDescriptor("enableHighQualityLighting")
	enableMotionBlur_ : EnableMotionBlurPlug = PlugDescriptor("enableMotionBlur")
	enableNonPowerOfTwoTexture_ : EnableNonPowerOfTwoTexturePlug = PlugDescriptor("enableNonPowerOfTwoTexture")
	frameBufferFormat_ : FrameBufferFormatPlug = PlugDescriptor("frameBufferFormat")
	graphicsHardwareGeometryCachingData_ : GraphicsHardwareGeometryCachingDataPlug = PlugDescriptor("graphicsHardwareGeometryCachingData")
	graphicsHardwareGeometryCachingIndexing_ : GraphicsHardwareGeometryCachingIndexingPlug = PlugDescriptor("graphicsHardwareGeometryCachingIndexing")
	hardwareCodec_ : HardwareCodecPlug = PlugDescriptor("hardwareCodec")
	hardwareDepth_ : HardwareDepthPlug = PlugDescriptor("hardwareDepth")
	hardwareEnvironmentLookup_ : HardwareEnvironmentLookupPlug = PlugDescriptor("hardwareEnvironmentLookup")
	hardwareFrameRate_ : HardwareFrameRatePlug = PlugDescriptor("hardwareFrameRate")
	hardwareQual_ : HardwareQualPlug = PlugDescriptor("hardwareQual")
	lightIntensityThreshold_ : LightIntensityThresholdPlug = PlugDescriptor("lightIntensityThreshold")
	maximumGeometryCacheSize_ : MaximumGeometryCacheSizePlug = PlugDescriptor("maximumGeometryCacheSize")
	motionBlurByFrame_ : MotionBlurByFramePlug = PlugDescriptor("motionBlurByFrame")
	numberOfExposures_ : NumberOfExposuresPlug = PlugDescriptor("numberOfExposures")
	numberOfSamples_ : NumberOfSamplesPlug = PlugDescriptor("numberOfSamples")
	shadingModel_ : ShadingModelPlug = PlugDescriptor("shadingModel")
	shadowsObeyLightLinking_ : ShadowsObeyLightLinkingPlug = PlugDescriptor("shadowsObeyLightLinking")
	shadowsObeyShadowLinking_ : ShadowsObeyShadowLinkingPlug = PlugDescriptor("shadowsObeyShadowLinking")
	smallObjectCulling_ : SmallObjectCullingPlug = PlugDescriptor("smallObjectCulling")
	textureCompression_ : TextureCompressionPlug = PlugDescriptor("textureCompression")
	transparencySorting_ : TransparencySortingPlug = PlugDescriptor("transparencySorting")
	transparentShadowCasting_ : TransparentShadowCastingPlug = PlugDescriptor("transparentShadowCasting")
	writeAlphaAsColor_ : WriteAlphaAsColorPlug = PlugDescriptor("writeAlphaAsColor")
	writeZDepthAsColor_ : WriteZDepthAsColorPlug = PlugDescriptor("writeZDepthAsColor")

	# node attributes

	typeName = "hardwareRenderGlobals"
	apiTypeInt = 527
	apiTypeStr = "kHardwareRenderGlobals"
	typeIdInt = 1213682247
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "blendSpecularWithAlpha", "bumpTextureResolution", "colorTextureResolution", "culling", "cullingThreshold", "enableAcceleratedMultiSampling", "enableEdgeAntiAliasing", "enableGeometryMask", "enableHighQualityLighting", "enableMotionBlur", "enableNonPowerOfTwoTexture", "frameBufferFormat", "graphicsHardwareGeometryCachingData", "graphicsHardwareGeometryCachingIndexing", "hardwareCodec", "hardwareDepth", "hardwareEnvironmentLookup", "hardwareFrameRate", "hardwareQual", "lightIntensityThreshold", "maximumGeometryCacheSize", "motionBlurByFrame", "numberOfExposures", "numberOfSamples", "shadingModel", "shadowsObeyLightLinking", "shadowsObeyShadowLinking", "smallObjectCulling", "textureCompression", "transparencySorting", "transparentShadowCasting", "writeAlphaAsColor", "writeZDepthAsColor"]
	nodeLeafPlugs = ["binMembership", "blendSpecularWithAlpha", "bumpTextureResolution", "colorTextureResolution", "culling", "cullingThreshold", "enableAcceleratedMultiSampling", "enableEdgeAntiAliasing", "enableGeometryMask", "enableHighQualityLighting", "enableMotionBlur", "enableNonPowerOfTwoTexture", "frameBufferFormat", "graphicsHardwareGeometryCachingData", "graphicsHardwareGeometryCachingIndexing", "hardwareCodec", "hardwareDepth", "hardwareEnvironmentLookup", "hardwareFrameRate", "hardwareQual", "lightIntensityThreshold", "maximumGeometryCacheSize", "motionBlurByFrame", "numberOfExposures", "numberOfSamples", "shadingModel", "shadowsObeyLightLinking", "shadowsObeyShadowLinking", "smallObjectCulling", "textureCompression", "transparencySorting", "transparentShadowCasting", "writeAlphaAsColor", "writeZDepthAsColor"]
	pass

