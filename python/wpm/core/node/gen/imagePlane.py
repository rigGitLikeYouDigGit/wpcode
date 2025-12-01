

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
class AlphaGainPlug(Plug):
	node : ImagePlane = None
	pass
class AlreadyPremultPlug(Plug):
	node : ImagePlane = None
	pass
class ColorGainBPlug(Plug):
	parent : ColorGainPlug = PlugDescriptor("colorGain")
	node : ImagePlane = None
	pass
class ColorGainGPlug(Plug):
	parent : ColorGainPlug = PlugDescriptor("colorGain")
	node : ImagePlane = None
	pass
class ColorGainRPlug(Plug):
	parent : ColorGainPlug = PlugDescriptor("colorGain")
	node : ImagePlane = None
	pass
class ColorGainPlug(Plug):
	colorGainB_ : ColorGainBPlug = PlugDescriptor("colorGainB")
	cgb_ : ColorGainBPlug = PlugDescriptor("colorGainB")
	colorGainG_ : ColorGainGPlug = PlugDescriptor("colorGainG")
	cgg_ : ColorGainGPlug = PlugDescriptor("colorGainG")
	colorGainR_ : ColorGainRPlug = PlugDescriptor("colorGainR")
	cgr_ : ColorGainRPlug = PlugDescriptor("colorGainR")
	node : ImagePlane = None
	pass
class ColorManagementConfigFileEnabledPlug(Plug):
	node : ImagePlane = None
	pass
class ColorManagementConfigFilePathPlug(Plug):
	node : ImagePlane = None
	pass
class ColorManagementEnabledPlug(Plug):
	node : ImagePlane = None
	pass
class ColorOffsetBPlug(Plug):
	parent : ColorOffsetPlug = PlugDescriptor("colorOffset")
	node : ImagePlane = None
	pass
class ColorOffsetGPlug(Plug):
	parent : ColorOffsetPlug = PlugDescriptor("colorOffset")
	node : ImagePlane = None
	pass
class ColorOffsetRPlug(Plug):
	parent : ColorOffsetPlug = PlugDescriptor("colorOffset")
	node : ImagePlane = None
	pass
class ColorOffsetPlug(Plug):
	colorOffsetB_ : ColorOffsetBPlug = PlugDescriptor("colorOffsetB")
	cob_ : ColorOffsetBPlug = PlugDescriptor("colorOffsetB")
	colorOffsetG_ : ColorOffsetGPlug = PlugDescriptor("colorOffsetG")
	cog_ : ColorOffsetGPlug = PlugDescriptor("colorOffsetG")
	colorOffsetR_ : ColorOffsetRPlug = PlugDescriptor("colorOffsetR")
	cor_ : ColorOffsetRPlug = PlugDescriptor("colorOffsetR")
	node : ImagePlane = None
	pass
class ColorSpacePlug(Plug):
	node : ImagePlane = None
	pass
class CompositeDepthPlug(Plug):
	node : ImagePlane = None
	pass
class CoverageXPlug(Plug):
	parent : CoveragePlug = PlugDescriptor("coverage")
	node : ImagePlane = None
	pass
class CoverageYPlug(Plug):
	parent : CoveragePlug = PlugDescriptor("coverage")
	node : ImagePlane = None
	pass
class CoveragePlug(Plug):
	coverageX_ : CoverageXPlug = PlugDescriptor("coverageX")
	cvx_ : CoverageXPlug = PlugDescriptor("coverageX")
	coverageY_ : CoverageYPlug = PlugDescriptor("coverageY")
	cvy_ : CoverageYPlug = PlugDescriptor("coverageY")
	node : ImagePlane = None
	pass
class CoverageOriginXPlug(Plug):
	parent : CoverageOriginPlug = PlugDescriptor("coverageOrigin")
	node : ImagePlane = None
	pass
class CoverageOriginYPlug(Plug):
	parent : CoverageOriginPlug = PlugDescriptor("coverageOrigin")
	node : ImagePlane = None
	pass
class CoverageOriginPlug(Plug):
	coverageOriginX_ : CoverageOriginXPlug = PlugDescriptor("coverageOriginX")
	cox_ : CoverageOriginXPlug = PlugDescriptor("coverageOriginX")
	coverageOriginY_ : CoverageOriginYPlug = PlugDescriptor("coverageOriginY")
	coy_ : CoverageOriginYPlug = PlugDescriptor("coverageOriginY")
	node : ImagePlane = None
	pass
class DepthPlug(Plug):
	node : ImagePlane = None
	pass
class DepthBiasPlug(Plug):
	node : ImagePlane = None
	pass
class DepthFilePlug(Plug):
	node : ImagePlane = None
	pass
class DepthOversamplePlug(Plug):
	node : ImagePlane = None
	pass
class DepthScalePlug(Plug):
	node : ImagePlane = None
	pass
class DisplayModePlug(Plug):
	node : ImagePlane = None
	pass
class DisplayOnlyIfCurrentPlug(Plug):
	node : ImagePlane = None
	pass
class FitPlug(Plug):
	node : ImagePlane = None
	pass
class FrameCachePlug(Plug):
	node : ImagePlane = None
	pass
class FrameExtensionPlug(Plug):
	node : ImagePlane = None
	pass
class FrameInPlug(Plug):
	node : ImagePlane = None
	pass
class FrameOffsetPlug(Plug):
	node : ImagePlane = None
	pass
class FrameOutPlug(Plug):
	node : ImagePlane = None
	pass
class FrameVisibilityPlug(Plug):
	node : ImagePlane = None
	pass
class HeightPlug(Plug):
	node : ImagePlane = None
	pass
class IgnoreColorSpaceFileRulesPlug(Plug):
	node : ImagePlane = None
	pass
class ImageCenterXPlug(Plug):
	parent : ImageCenterPlug = PlugDescriptor("imageCenter")
	node : ImagePlane = None
	pass
class ImageCenterYPlug(Plug):
	parent : ImageCenterPlug = PlugDescriptor("imageCenter")
	node : ImagePlane = None
	pass
class ImageCenterZPlug(Plug):
	parent : ImageCenterPlug = PlugDescriptor("imageCenter")
	node : ImagePlane = None
	pass
class ImageCenterPlug(Plug):
	imageCenterX_ : ImageCenterXPlug = PlugDescriptor("imageCenterX")
	icx_ : ImageCenterXPlug = PlugDescriptor("imageCenterX")
	imageCenterY_ : ImageCenterYPlug = PlugDescriptor("imageCenterY")
	icy_ : ImageCenterYPlug = PlugDescriptor("imageCenterY")
	imageCenterZ_ : ImageCenterZPlug = PlugDescriptor("imageCenterZ")
	icz_ : ImageCenterZPlug = PlugDescriptor("imageCenterZ")
	node : ImagePlane = None
	pass
class ImageNamePlug(Plug):
	node : ImagePlane = None
	pass
class LockedToCameraPlug(Plug):
	node : ImagePlane = None
	pass
class LookThroughCameraPlug(Plug):
	node : ImagePlane = None
	pass
class MaintainRatioPlug(Plug):
	node : ImagePlane = None
	pass
class MaxShadingSamplesPlug(Plug):
	node : ImagePlane = None
	pass
class OffsetXPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : ImagePlane = None
	pass
class OffsetYPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : ImagePlane = None
	pass
class OffsetPlug(Plug):
	offsetX_ : OffsetXPlug = PlugDescriptor("offsetX")
	ox_ : OffsetXPlug = PlugDescriptor("offsetX")
	offsetY_ : OffsetYPlug = PlugDescriptor("offsetY")
	oy_ : OffsetYPlug = PlugDescriptor("offsetY")
	node : ImagePlane = None
	pass
class OutputFrameExtensionPlug(Plug):
	node : ImagePlane = None
	pass
class OutputImagePlug(Plug):
	node : ImagePlane = None
	pass
class OutputImageFramesPlug(Plug):
	parent : OutputImageDimensionsPlug = PlugDescriptor("outputImageDimensions")
	node : ImagePlane = None
	pass
class OutputImageHeightPlug(Plug):
	parent : OutputImageDimensionsPlug = PlugDescriptor("outputImageDimensions")
	node : ImagePlane = None
	pass
class OutputImageWidthPlug(Plug):
	parent : OutputImageDimensionsPlug = PlugDescriptor("outputImageDimensions")
	node : ImagePlane = None
	pass
class OutputImageDimensionsPlug(Plug):
	outputImageFrames_ : OutputImageFramesPlug = PlugDescriptor("outputImageFrames")
	oif_ : OutputImageFramesPlug = PlugDescriptor("outputImageFrames")
	outputImageHeight_ : OutputImageHeightPlug = PlugDescriptor("outputImageHeight")
	oih_ : OutputImageHeightPlug = PlugDescriptor("outputImageHeight")
	outputImageWidth_ : OutputImageWidthPlug = PlugDescriptor("outputImageWidth")
	oiw_ : OutputImageWidthPlug = PlugDescriptor("outputImageWidth")
	node : ImagePlane = None
	pass
class OutputImageFlagsPlug(Plug):
	node : ImagePlane = None
	pass
class ResolvedFilePathPlug(Plug):
	node : ImagePlane = None
	pass
class RotatePlug(Plug):
	node : ImagePlane = None
	pass
class SeparateDepthPlug(Plug):
	node : ImagePlane = None
	pass
class ShadingSamplesPlug(Plug):
	node : ImagePlane = None
	pass
class ShadingSamplesOverridePlug(Plug):
	node : ImagePlane = None
	pass
class SizeXPlug(Plug):
	parent : SizePlug = PlugDescriptor("size")
	node : ImagePlane = None
	pass
class SizeYPlug(Plug):
	parent : SizePlug = PlugDescriptor("size")
	node : ImagePlane = None
	pass
class SizePlug(Plug):
	sizeX_ : SizeXPlug = PlugDescriptor("sizeX")
	sx_ : SizeXPlug = PlugDescriptor("sizeX")
	sizeY_ : SizeYPlug = PlugDescriptor("sizeY")
	sy_ : SizeYPlug = PlugDescriptor("sizeY")
	node : ImagePlane = None
	pass
class SourceTexturePlug(Plug):
	node : ImagePlane = None
	pass
class SqueezeCorrectionPlug(Plug):
	node : ImagePlane = None
	pass
class TextureFilterPlug(Plug):
	node : ImagePlane = None
	pass
class TypePlug(Plug):
	node : ImagePlane = None
	pass
class UseDepthMapPlug(Plug):
	node : ImagePlane = None
	pass
class UseFrameExtensionPlug(Plug):
	node : ImagePlane = None
	pass
class ViewNameStrPlug(Plug):
	node : ImagePlane = None
	pass
class ViewNameUsedPlug(Plug):
	node : ImagePlane = None
	pass
class VisibleInReflectionsPlug(Plug):
	node : ImagePlane = None
	pass
class VisibleInRefractionsPlug(Plug):
	node : ImagePlane = None
	pass
class WidthPlug(Plug):
	node : ImagePlane = None
	pass
class WorkingSpacePlug(Plug):
	node : ImagePlane = None
	pass
# endregion


# define node class
class ImagePlane(Shape):
	alphaGain_ : AlphaGainPlug = PlugDescriptor("alphaGain")
	alreadyPremult_ : AlreadyPremultPlug = PlugDescriptor("alreadyPremult")
	colorGainB_ : ColorGainBPlug = PlugDescriptor("colorGainB")
	colorGainG_ : ColorGainGPlug = PlugDescriptor("colorGainG")
	colorGainR_ : ColorGainRPlug = PlugDescriptor("colorGainR")
	colorGain_ : ColorGainPlug = PlugDescriptor("colorGain")
	colorManagementConfigFileEnabled_ : ColorManagementConfigFileEnabledPlug = PlugDescriptor("colorManagementConfigFileEnabled")
	colorManagementConfigFilePath_ : ColorManagementConfigFilePathPlug = PlugDescriptor("colorManagementConfigFilePath")
	colorManagementEnabled_ : ColorManagementEnabledPlug = PlugDescriptor("colorManagementEnabled")
	colorOffsetB_ : ColorOffsetBPlug = PlugDescriptor("colorOffsetB")
	colorOffsetG_ : ColorOffsetGPlug = PlugDescriptor("colorOffsetG")
	colorOffsetR_ : ColorOffsetRPlug = PlugDescriptor("colorOffsetR")
	colorOffset_ : ColorOffsetPlug = PlugDescriptor("colorOffset")
	colorSpace_ : ColorSpacePlug = PlugDescriptor("colorSpace")
	compositeDepth_ : CompositeDepthPlug = PlugDescriptor("compositeDepth")
	coverageX_ : CoverageXPlug = PlugDescriptor("coverageX")
	coverageY_ : CoverageYPlug = PlugDescriptor("coverageY")
	coverage_ : CoveragePlug = PlugDescriptor("coverage")
	coverageOriginX_ : CoverageOriginXPlug = PlugDescriptor("coverageOriginX")
	coverageOriginY_ : CoverageOriginYPlug = PlugDescriptor("coverageOriginY")
	coverageOrigin_ : CoverageOriginPlug = PlugDescriptor("coverageOrigin")
	depth_ : DepthPlug = PlugDescriptor("depth")
	depthBias_ : DepthBiasPlug = PlugDescriptor("depthBias")
	depthFile_ : DepthFilePlug = PlugDescriptor("depthFile")
	depthOversample_ : DepthOversamplePlug = PlugDescriptor("depthOversample")
	depthScale_ : DepthScalePlug = PlugDescriptor("depthScale")
	displayMode_ : DisplayModePlug = PlugDescriptor("displayMode")
	displayOnlyIfCurrent_ : DisplayOnlyIfCurrentPlug = PlugDescriptor("displayOnlyIfCurrent")
	fit_ : FitPlug = PlugDescriptor("fit")
	frameCache_ : FrameCachePlug = PlugDescriptor("frameCache")
	frameExtension_ : FrameExtensionPlug = PlugDescriptor("frameExtension")
	frameIn_ : FrameInPlug = PlugDescriptor("frameIn")
	frameOffset_ : FrameOffsetPlug = PlugDescriptor("frameOffset")
	frameOut_ : FrameOutPlug = PlugDescriptor("frameOut")
	frameVisibility_ : FrameVisibilityPlug = PlugDescriptor("frameVisibility")
	height_ : HeightPlug = PlugDescriptor("height")
	ignoreColorSpaceFileRules_ : IgnoreColorSpaceFileRulesPlug = PlugDescriptor("ignoreColorSpaceFileRules")
	imageCenterX_ : ImageCenterXPlug = PlugDescriptor("imageCenterX")
	imageCenterY_ : ImageCenterYPlug = PlugDescriptor("imageCenterY")
	imageCenterZ_ : ImageCenterZPlug = PlugDescriptor("imageCenterZ")
	imageCenter_ : ImageCenterPlug = PlugDescriptor("imageCenter")
	imageName_ : ImageNamePlug = PlugDescriptor("imageName")
	lockedToCamera_ : LockedToCameraPlug = PlugDescriptor("lockedToCamera")
	lookThroughCamera_ : LookThroughCameraPlug = PlugDescriptor("lookThroughCamera")
	maintainRatio_ : MaintainRatioPlug = PlugDescriptor("maintainRatio")
	maxShadingSamples_ : MaxShadingSamplesPlug = PlugDescriptor("maxShadingSamples")
	offsetX_ : OffsetXPlug = PlugDescriptor("offsetX")
	offsetY_ : OffsetYPlug = PlugDescriptor("offsetY")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	outputFrameExtension_ : OutputFrameExtensionPlug = PlugDescriptor("outputFrameExtension")
	outputImage_ : OutputImagePlug = PlugDescriptor("outputImage")
	outputImageFrames_ : OutputImageFramesPlug = PlugDescriptor("outputImageFrames")
	outputImageHeight_ : OutputImageHeightPlug = PlugDescriptor("outputImageHeight")
	outputImageWidth_ : OutputImageWidthPlug = PlugDescriptor("outputImageWidth")
	outputImageDimensions_ : OutputImageDimensionsPlug = PlugDescriptor("outputImageDimensions")
	outputImageFlags_ : OutputImageFlagsPlug = PlugDescriptor("outputImageFlags")
	resolvedFilePath_ : ResolvedFilePathPlug = PlugDescriptor("resolvedFilePath")
	rotate_ : RotatePlug = PlugDescriptor("rotate")
	separateDepth_ : SeparateDepthPlug = PlugDescriptor("separateDepth")
	shadingSamples_ : ShadingSamplesPlug = PlugDescriptor("shadingSamples")
	shadingSamplesOverride_ : ShadingSamplesOverridePlug = PlugDescriptor("shadingSamplesOverride")
	sizeX_ : SizeXPlug = PlugDescriptor("sizeX")
	sizeY_ : SizeYPlug = PlugDescriptor("sizeY")
	size_ : SizePlug = PlugDescriptor("size")
	sourceTexture_ : SourceTexturePlug = PlugDescriptor("sourceTexture")
	squeezeCorrection_ : SqueezeCorrectionPlug = PlugDescriptor("squeezeCorrection")
	textureFilter_ : TextureFilterPlug = PlugDescriptor("textureFilter")
	type_ : TypePlug = PlugDescriptor("type")
	useDepthMap_ : UseDepthMapPlug = PlugDescriptor("useDepthMap")
	useFrameExtension_ : UseFrameExtensionPlug = PlugDescriptor("useFrameExtension")
	viewNameStr_ : ViewNameStrPlug = PlugDescriptor("viewNameStr")
	viewNameUsed_ : ViewNameUsedPlug = PlugDescriptor("viewNameUsed")
	visibleInReflections_ : VisibleInReflectionsPlug = PlugDescriptor("visibleInReflections")
	visibleInRefractions_ : VisibleInRefractionsPlug = PlugDescriptor("visibleInRefractions")
	width_ : WidthPlug = PlugDescriptor("width")
	workingSpace_ : WorkingSpacePlug = PlugDescriptor("workingSpace")

	# node attributes

	typeName = "imagePlane"
	apiTypeInt = 370
	apiTypeStr = "kImagePlane"
	typeIdInt = 1145655372
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["alphaGain", "alreadyPremult", "colorGainB", "colorGainG", "colorGainR", "colorGain", "colorManagementConfigFileEnabled", "colorManagementConfigFilePath", "colorManagementEnabled", "colorOffsetB", "colorOffsetG", "colorOffsetR", "colorOffset", "colorSpace", "compositeDepth", "coverageX", "coverageY", "coverage", "coverageOriginX", "coverageOriginY", "coverageOrigin", "depth", "depthBias", "depthFile", "depthOversample", "depthScale", "displayMode", "displayOnlyIfCurrent", "fit", "frameCache", "frameExtension", "frameIn", "frameOffset", "frameOut", "frameVisibility", "height", "ignoreColorSpaceFileRules", "imageCenterX", "imageCenterY", "imageCenterZ", "imageCenter", "imageName", "lockedToCamera", "lookThroughCamera", "maintainRatio", "maxShadingSamples", "offsetX", "offsetY", "offset", "outputFrameExtension", "outputImage", "outputImageFrames", "outputImageHeight", "outputImageWidth", "outputImageDimensions", "outputImageFlags", "resolvedFilePath", "rotate", "separateDepth", "shadingSamples", "shadingSamplesOverride", "sizeX", "sizeY", "size", "sourceTexture", "squeezeCorrection", "textureFilter", "type", "useDepthMap", "useFrameExtension", "viewNameStr", "viewNameUsed", "visibleInReflections", "visibleInRefractions", "width", "workingSpace"]
	nodeLeafPlugs = ["alphaGain", "alreadyPremult", "colorGain", "colorManagementConfigFileEnabled", "colorManagementConfigFilePath", "colorManagementEnabled", "colorOffset", "colorSpace", "compositeDepth", "coverage", "coverageOrigin", "depth", "depthBias", "depthFile", "depthOversample", "depthScale", "displayMode", "displayOnlyIfCurrent", "fit", "frameCache", "frameExtension", "frameIn", "frameOffset", "frameOut", "frameVisibility", "height", "ignoreColorSpaceFileRules", "imageCenter", "imageName", "lockedToCamera", "lookThroughCamera", "maintainRatio", "maxShadingSamples", "offset", "outputFrameExtension", "outputImage", "outputImageDimensions", "outputImageFlags", "resolvedFilePath", "rotate", "separateDepth", "shadingSamples", "shadingSamplesOverride", "size", "sourceTexture", "squeezeCorrection", "textureFilter", "type", "useDepthMap", "useFrameExtension", "viewNameStr", "viewNameUsed", "visibleInReflections", "visibleInRefractions", "width", "workingSpace"]
	pass

