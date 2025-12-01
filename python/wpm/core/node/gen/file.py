

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Texture2d = retriever.getNodeCls("Texture2d")
assert Texture2d
if T.TYPE_CHECKING:
	from .. import Texture2d

# add node doc



# region plug type defs
class BaseExplicitUvTilePositionUPlug(Plug):
	parent : BaseExplicitUvTilePositionPlug = PlugDescriptor("baseExplicitUvTilePosition")
	node : File = None
	pass
class BaseExplicitUvTilePositionVPlug(Plug):
	parent : BaseExplicitUvTilePositionPlug = PlugDescriptor("baseExplicitUvTilePosition")
	node : File = None
	pass
class BaseExplicitUvTilePositionPlug(Plug):
	baseExplicitUvTilePositionU_ : BaseExplicitUvTilePositionUPlug = PlugDescriptor("baseExplicitUvTilePositionU")
	bupu_ : BaseExplicitUvTilePositionUPlug = PlugDescriptor("baseExplicitUvTilePositionU")
	baseExplicitUvTilePositionV_ : BaseExplicitUvTilePositionVPlug = PlugDescriptor("baseExplicitUvTilePositionV")
	bupv_ : BaseExplicitUvTilePositionVPlug = PlugDescriptor("baseExplicitUvTilePositionV")
	node : File = None
	pass
class BlurPixelationPlug(Plug):
	node : File = None
	pass
class ByCycleIncrementPlug(Plug):
	node : File = None
	pass
class ColorManagementConfigFileEnabledPlug(Plug):
	node : File = None
	pass
class ColorManagementConfigFilePathPlug(Plug):
	node : File = None
	pass
class ColorManagementEnabledPlug(Plug):
	node : File = None
	pass
class ColorProfilePlug(Plug):
	node : File = None
	pass
class ColorSpacePlug(Plug):
	node : File = None
	pass
class ComputedFileTextureNamePatternPlug(Plug):
	node : File = None
	pass
class CoverageUPlug(Plug):
	parent : CoveragePlug = PlugDescriptor("coverage")
	node : File = None
	pass
class CoverageVPlug(Plug):
	parent : CoveragePlug = PlugDescriptor("coverage")
	node : File = None
	pass
class CoveragePlug(Plug):
	coverageU_ : CoverageUPlug = PlugDescriptor("coverageU")
	cu_ : CoverageUPlug = PlugDescriptor("coverageU")
	coverageV_ : CoverageVPlug = PlugDescriptor("coverageV")
	cv_ : CoverageVPlug = PlugDescriptor("coverageV")
	node : File = None
	pass
class DirtyPixelRegionPlug(Plug):
	node : File = None
	pass
class DisableFileLoadPlug(Plug):
	node : File = None
	pass
class DoTransformPlug(Plug):
	node : File = None
	pass
class EndCycleExtensionPlug(Plug):
	node : File = None
	pass
class ExplicitUvTileNamePlug(Plug):
	parent : ExplicitUvTilesPlug = PlugDescriptor("explicitUvTiles")
	node : File = None
	pass
class ExplicitUvTilePositionUPlug(Plug):
	parent : ExplicitUvTilePositionPlug = PlugDescriptor("explicitUvTilePosition")
	node : File = None
	pass
class ExplicitUvTilePositionVPlug(Plug):
	parent : ExplicitUvTilePositionPlug = PlugDescriptor("explicitUvTilePosition")
	node : File = None
	pass
class ExplicitUvTilePositionPlug(Plug):
	parent : ExplicitUvTilesPlug = PlugDescriptor("explicitUvTiles")
	explicitUvTilePositionU_ : ExplicitUvTilePositionUPlug = PlugDescriptor("explicitUvTilePositionU")
	eupu_ : ExplicitUvTilePositionUPlug = PlugDescriptor("explicitUvTilePositionU")
	explicitUvTilePositionV_ : ExplicitUvTilePositionVPlug = PlugDescriptor("explicitUvTilePositionV")
	eupv_ : ExplicitUvTilePositionVPlug = PlugDescriptor("explicitUvTilePositionV")
	node : File = None
	pass
class ExplicitUvTilesPlug(Plug):
	explicitUvTileName_ : ExplicitUvTileNamePlug = PlugDescriptor("explicitUvTileName")
	eutn_ : ExplicitUvTileNamePlug = PlugDescriptor("explicitUvTileName")
	explicitUvTilePosition_ : ExplicitUvTilePositionPlug = PlugDescriptor("explicitUvTilePosition")
	eutp_ : ExplicitUvTilePositionPlug = PlugDescriptor("explicitUvTilePosition")
	node : File = None
	pass
class ExposurePlug(Plug):
	node : File = None
	pass
class FileHasAlphaPlug(Plug):
	node : File = None
	pass
class FileTextureNamePlug(Plug):
	node : File = None
	pass
class FileTextureNamePatternPlug(Plug):
	node : File = None
	pass
class FilterTypePlug(Plug):
	node : File = None
	pass
class FilterWidthPlug(Plug):
	node : File = None
	pass
class ForceSwatchGenPlug(Plug):
	node : File = None
	pass
class FrameExtensionPlug(Plug):
	node : File = None
	pass
class FrameOffsetPlug(Plug):
	node : File = None
	pass
class HdrExposurePlug(Plug):
	node : File = None
	pass
class HdrMappingPlug(Plug):
	node : File = None
	pass
class IgnoreColorSpaceFileRulesPlug(Plug):
	node : File = None
	pass
class InfoBitsPlug(Plug):
	node : File = None
	pass
class MirrorUPlug(Plug):
	node : File = None
	pass
class MirrorVPlug(Plug):
	node : File = None
	pass
class NoiseUPlug(Plug):
	parent : NoiseUVPlug = PlugDescriptor("noiseUV")
	node : File = None
	pass
class NoiseVPlug(Plug):
	parent : NoiseUVPlug = PlugDescriptor("noiseUV")
	node : File = None
	pass
class NoiseUVPlug(Plug):
	noiseU_ : NoiseUPlug = PlugDescriptor("noiseU")
	nu_ : NoiseUPlug = PlugDescriptor("noiseU")
	noiseV_ : NoiseVPlug = PlugDescriptor("noiseV")
	nv_ : NoiseVPlug = PlugDescriptor("noiseV")
	node : File = None
	pass
class ObjectTypePlug(Plug):
	node : File = None
	pass
class OffsetUPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : File = None
	pass
class OffsetVPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : File = None
	pass
class OffsetPlug(Plug):
	offsetU_ : OffsetUPlug = PlugDescriptor("offsetU")
	ofu_ : OffsetUPlug = PlugDescriptor("offsetU")
	offsetV_ : OffsetVPlug = PlugDescriptor("offsetV")
	ofv_ : OffsetVPlug = PlugDescriptor("offsetV")
	node : File = None
	pass
class OutSizeXPlug(Plug):
	parent : OutSizePlug = PlugDescriptor("outSize")
	node : File = None
	pass
class OutSizeYPlug(Plug):
	parent : OutSizePlug = PlugDescriptor("outSize")
	node : File = None
	pass
class OutSizePlug(Plug):
	outSizeX_ : OutSizeXPlug = PlugDescriptor("outSizeX")
	osx_ : OutSizeXPlug = PlugDescriptor("outSizeX")
	outSizeY_ : OutSizeYPlug = PlugDescriptor("outSizeY")
	osy_ : OutSizeYPlug = PlugDescriptor("outSizeY")
	node : File = None
	pass
class OutTransparencyBPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : File = None
	pass
class OutTransparencyGPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : File = None
	pass
class OutTransparencyRPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : File = None
	pass
class OutTransparencyPlug(Plug):
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	otb_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	otg_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	otr_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	node : File = None
	pass
class PixelCenterXPlug(Plug):
	parent : PixelCenterPlug = PlugDescriptor("pixelCenter")
	node : File = None
	pass
class PixelCenterYPlug(Plug):
	parent : PixelCenterPlug = PlugDescriptor("pixelCenter")
	node : File = None
	pass
class PixelCenterPlug(Plug):
	pixelCenterX_ : PixelCenterXPlug = PlugDescriptor("pixelCenterX")
	pcx_ : PixelCenterXPlug = PlugDescriptor("pixelCenterX")
	pixelCenterY_ : PixelCenterYPlug = PlugDescriptor("pixelCenterY")
	pcy_ : PixelCenterYPlug = PlugDescriptor("pixelCenterY")
	node : File = None
	pass
class PreFilterPlug(Plug):
	node : File = None
	pass
class PreFilterRadiusPlug(Plug):
	node : File = None
	pass
class PrimitiveIdPlug(Plug):
	node : File = None
	pass
class PtexFilterBlurPlug(Plug):
	node : File = None
	pass
class PtexFilterInterpolateLevelsPlug(Plug):
	node : File = None
	pass
class PtexFilterSharpnessPlug(Plug):
	node : File = None
	pass
class PtexFilterTypePlug(Plug):
	node : File = None
	pass
class PtexFilterWidthPlug(Plug):
	node : File = None
	pass
class RayDepthPlug(Plug):
	node : File = None
	pass
class RepeatUPlug(Plug):
	parent : RepeatUVPlug = PlugDescriptor("repeatUV")
	node : File = None
	pass
class RepeatVPlug(Plug):
	parent : RepeatUVPlug = PlugDescriptor("repeatUV")
	node : File = None
	pass
class RepeatUVPlug(Plug):
	repeatU_ : RepeatUPlug = PlugDescriptor("repeatU")
	reu_ : RepeatUPlug = PlugDescriptor("repeatU")
	repeatV_ : RepeatVPlug = PlugDescriptor("repeatV")
	rev_ : RepeatVPlug = PlugDescriptor("repeatV")
	node : File = None
	pass
class RotateFramePlug(Plug):
	node : File = None
	pass
class RotateUVPlug(Plug):
	node : File = None
	pass
class StaggerPlug(Plug):
	node : File = None
	pass
class StartCycleExtensionPlug(Plug):
	node : File = None
	pass
class TranslateFrameUPlug(Plug):
	parent : TranslateFramePlug = PlugDescriptor("translateFrame")
	node : File = None
	pass
class TranslateFrameVPlug(Plug):
	parent : TranslateFramePlug = PlugDescriptor("translateFrame")
	node : File = None
	pass
class TranslateFramePlug(Plug):
	translateFrameU_ : TranslateFrameUPlug = PlugDescriptor("translateFrameU")
	tfu_ : TranslateFrameUPlug = PlugDescriptor("translateFrameU")
	translateFrameV_ : TranslateFrameVPlug = PlugDescriptor("translateFrameV")
	tfv_ : TranslateFrameVPlug = PlugDescriptor("translateFrameV")
	node : File = None
	pass
class UseCachePlug(Plug):
	node : File = None
	pass
class UseFrameExtensionPlug(Plug):
	node : File = None
	pass
class UseHardwareTextureCyclingPlug(Plug):
	node : File = None
	pass
class UseMaximumResPlug(Plug):
	node : File = None
	pass
class UvTileProxyDirtyPlug(Plug):
	node : File = None
	pass
class UvTileProxyGeneratePlug(Plug):
	node : File = None
	pass
class UvTileProxyQualityPlug(Plug):
	node : File = None
	pass
class UvTilingModePlug(Plug):
	node : File = None
	pass
class VertexCameraOneXPlug(Plug):
	parent : VertexCameraOnePlug = PlugDescriptor("vertexCameraOne")
	node : File = None
	pass
class VertexCameraOneYPlug(Plug):
	parent : VertexCameraOnePlug = PlugDescriptor("vertexCameraOne")
	node : File = None
	pass
class VertexCameraOneZPlug(Plug):
	parent : VertexCameraOnePlug = PlugDescriptor("vertexCameraOne")
	node : File = None
	pass
class VertexCameraOnePlug(Plug):
	vertexCameraOneX_ : VertexCameraOneXPlug = PlugDescriptor("vertexCameraOneX")
	c1x_ : VertexCameraOneXPlug = PlugDescriptor("vertexCameraOneX")
	vertexCameraOneY_ : VertexCameraOneYPlug = PlugDescriptor("vertexCameraOneY")
	c1y_ : VertexCameraOneYPlug = PlugDescriptor("vertexCameraOneY")
	vertexCameraOneZ_ : VertexCameraOneZPlug = PlugDescriptor("vertexCameraOneZ")
	c1z_ : VertexCameraOneZPlug = PlugDescriptor("vertexCameraOneZ")
	node : File = None
	pass
class VertexCameraThreeXPlug(Plug):
	parent : VertexCameraThreePlug = PlugDescriptor("vertexCameraThree")
	node : File = None
	pass
class VertexCameraThreeYPlug(Plug):
	parent : VertexCameraThreePlug = PlugDescriptor("vertexCameraThree")
	node : File = None
	pass
class VertexCameraThreeZPlug(Plug):
	parent : VertexCameraThreePlug = PlugDescriptor("vertexCameraThree")
	node : File = None
	pass
class VertexCameraThreePlug(Plug):
	vertexCameraThreeX_ : VertexCameraThreeXPlug = PlugDescriptor("vertexCameraThreeX")
	c3x_ : VertexCameraThreeXPlug = PlugDescriptor("vertexCameraThreeX")
	vertexCameraThreeY_ : VertexCameraThreeYPlug = PlugDescriptor("vertexCameraThreeY")
	c3y_ : VertexCameraThreeYPlug = PlugDescriptor("vertexCameraThreeY")
	vertexCameraThreeZ_ : VertexCameraThreeZPlug = PlugDescriptor("vertexCameraThreeZ")
	c3z_ : VertexCameraThreeZPlug = PlugDescriptor("vertexCameraThreeZ")
	node : File = None
	pass
class VertexCameraTwoXPlug(Plug):
	parent : VertexCameraTwoPlug = PlugDescriptor("vertexCameraTwo")
	node : File = None
	pass
class VertexCameraTwoYPlug(Plug):
	parent : VertexCameraTwoPlug = PlugDescriptor("vertexCameraTwo")
	node : File = None
	pass
class VertexCameraTwoZPlug(Plug):
	parent : VertexCameraTwoPlug = PlugDescriptor("vertexCameraTwo")
	node : File = None
	pass
class VertexCameraTwoPlug(Plug):
	vertexCameraTwoX_ : VertexCameraTwoXPlug = PlugDescriptor("vertexCameraTwoX")
	c2x_ : VertexCameraTwoXPlug = PlugDescriptor("vertexCameraTwoX")
	vertexCameraTwoY_ : VertexCameraTwoYPlug = PlugDescriptor("vertexCameraTwoY")
	c2y_ : VertexCameraTwoYPlug = PlugDescriptor("vertexCameraTwoY")
	vertexCameraTwoZ_ : VertexCameraTwoZPlug = PlugDescriptor("vertexCameraTwoZ")
	c2z_ : VertexCameraTwoZPlug = PlugDescriptor("vertexCameraTwoZ")
	node : File = None
	pass
class VertexUvOneUPlug(Plug):
	parent : VertexUvOnePlug = PlugDescriptor("vertexUvOne")
	node : File = None
	pass
class VertexUvOneVPlug(Plug):
	parent : VertexUvOnePlug = PlugDescriptor("vertexUvOne")
	node : File = None
	pass
class VertexUvOnePlug(Plug):
	vertexUvOneU_ : VertexUvOneUPlug = PlugDescriptor("vertexUvOneU")
	t1u_ : VertexUvOneUPlug = PlugDescriptor("vertexUvOneU")
	vertexUvOneV_ : VertexUvOneVPlug = PlugDescriptor("vertexUvOneV")
	t1v_ : VertexUvOneVPlug = PlugDescriptor("vertexUvOneV")
	node : File = None
	pass
class VertexUvThreeUPlug(Plug):
	parent : VertexUvThreePlug = PlugDescriptor("vertexUvThree")
	node : File = None
	pass
class VertexUvThreeVPlug(Plug):
	parent : VertexUvThreePlug = PlugDescriptor("vertexUvThree")
	node : File = None
	pass
class VertexUvThreePlug(Plug):
	vertexUvThreeU_ : VertexUvThreeUPlug = PlugDescriptor("vertexUvThreeU")
	t3u_ : VertexUvThreeUPlug = PlugDescriptor("vertexUvThreeU")
	vertexUvThreeV_ : VertexUvThreeVPlug = PlugDescriptor("vertexUvThreeV")
	t3v_ : VertexUvThreeVPlug = PlugDescriptor("vertexUvThreeV")
	node : File = None
	pass
class VertexUvTwoUPlug(Plug):
	parent : VertexUvTwoPlug = PlugDescriptor("vertexUvTwo")
	node : File = None
	pass
class VertexUvTwoVPlug(Plug):
	parent : VertexUvTwoPlug = PlugDescriptor("vertexUvTwo")
	node : File = None
	pass
class VertexUvTwoPlug(Plug):
	vertexUvTwoU_ : VertexUvTwoUPlug = PlugDescriptor("vertexUvTwoU")
	t2u_ : VertexUvTwoUPlug = PlugDescriptor("vertexUvTwoU")
	vertexUvTwoV_ : VertexUvTwoVPlug = PlugDescriptor("vertexUvTwoV")
	t2v_ : VertexUvTwoVPlug = PlugDescriptor("vertexUvTwoV")
	node : File = None
	pass
class ViewNameStrPlug(Plug):
	node : File = None
	pass
class ViewNameUsedPlug(Plug):
	node : File = None
	pass
class WorkingSpacePlug(Plug):
	node : File = None
	pass
class WrapUPlug(Plug):
	node : File = None
	pass
class WrapVPlug(Plug):
	node : File = None
	pass
# endregion


# define node class
class File(Texture2d):
	baseExplicitUvTilePositionU_ : BaseExplicitUvTilePositionUPlug = PlugDescriptor("baseExplicitUvTilePositionU")
	baseExplicitUvTilePositionV_ : BaseExplicitUvTilePositionVPlug = PlugDescriptor("baseExplicitUvTilePositionV")
	baseExplicitUvTilePosition_ : BaseExplicitUvTilePositionPlug = PlugDescriptor("baseExplicitUvTilePosition")
	blurPixelation_ : BlurPixelationPlug = PlugDescriptor("blurPixelation")
	byCycleIncrement_ : ByCycleIncrementPlug = PlugDescriptor("byCycleIncrement")
	colorManagementConfigFileEnabled_ : ColorManagementConfigFileEnabledPlug = PlugDescriptor("colorManagementConfigFileEnabled")
	colorManagementConfigFilePath_ : ColorManagementConfigFilePathPlug = PlugDescriptor("colorManagementConfigFilePath")
	colorManagementEnabled_ : ColorManagementEnabledPlug = PlugDescriptor("colorManagementEnabled")
	colorProfile_ : ColorProfilePlug = PlugDescriptor("colorProfile")
	colorSpace_ : ColorSpacePlug = PlugDescriptor("colorSpace")
	computedFileTextureNamePattern_ : ComputedFileTextureNamePatternPlug = PlugDescriptor("computedFileTextureNamePattern")
	coverageU_ : CoverageUPlug = PlugDescriptor("coverageU")
	coverageV_ : CoverageVPlug = PlugDescriptor("coverageV")
	coverage_ : CoveragePlug = PlugDescriptor("coverage")
	dirtyPixelRegion_ : DirtyPixelRegionPlug = PlugDescriptor("dirtyPixelRegion")
	disableFileLoad_ : DisableFileLoadPlug = PlugDescriptor("disableFileLoad")
	doTransform_ : DoTransformPlug = PlugDescriptor("doTransform")
	endCycleExtension_ : EndCycleExtensionPlug = PlugDescriptor("endCycleExtension")
	explicitUvTileName_ : ExplicitUvTileNamePlug = PlugDescriptor("explicitUvTileName")
	explicitUvTilePositionU_ : ExplicitUvTilePositionUPlug = PlugDescriptor("explicitUvTilePositionU")
	explicitUvTilePositionV_ : ExplicitUvTilePositionVPlug = PlugDescriptor("explicitUvTilePositionV")
	explicitUvTilePosition_ : ExplicitUvTilePositionPlug = PlugDescriptor("explicitUvTilePosition")
	explicitUvTiles_ : ExplicitUvTilesPlug = PlugDescriptor("explicitUvTiles")
	exposure_ : ExposurePlug = PlugDescriptor("exposure")
	fileHasAlpha_ : FileHasAlphaPlug = PlugDescriptor("fileHasAlpha")
	fileTextureName_ : FileTextureNamePlug = PlugDescriptor("fileTextureName")
	fileTextureNamePattern_ : FileTextureNamePatternPlug = PlugDescriptor("fileTextureNamePattern")
	filterType_ : FilterTypePlug = PlugDescriptor("filterType")
	filterWidth_ : FilterWidthPlug = PlugDescriptor("filterWidth")
	forceSwatchGen_ : ForceSwatchGenPlug = PlugDescriptor("forceSwatchGen")
	frameExtension_ : FrameExtensionPlug = PlugDescriptor("frameExtension")
	frameOffset_ : FrameOffsetPlug = PlugDescriptor("frameOffset")
	hdrExposure_ : HdrExposurePlug = PlugDescriptor("hdrExposure")
	hdrMapping_ : HdrMappingPlug = PlugDescriptor("hdrMapping")
	ignoreColorSpaceFileRules_ : IgnoreColorSpaceFileRulesPlug = PlugDescriptor("ignoreColorSpaceFileRules")
	infoBits_ : InfoBitsPlug = PlugDescriptor("infoBits")
	mirrorU_ : MirrorUPlug = PlugDescriptor("mirrorU")
	mirrorV_ : MirrorVPlug = PlugDescriptor("mirrorV")
	noiseU_ : NoiseUPlug = PlugDescriptor("noiseU")
	noiseV_ : NoiseVPlug = PlugDescriptor("noiseV")
	noiseUV_ : NoiseUVPlug = PlugDescriptor("noiseUV")
	objectType_ : ObjectTypePlug = PlugDescriptor("objectType")
	offsetU_ : OffsetUPlug = PlugDescriptor("offsetU")
	offsetV_ : OffsetVPlug = PlugDescriptor("offsetV")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	outSizeX_ : OutSizeXPlug = PlugDescriptor("outSizeX")
	outSizeY_ : OutSizeYPlug = PlugDescriptor("outSizeY")
	outSize_ : OutSizePlug = PlugDescriptor("outSize")
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	outTransparency_ : OutTransparencyPlug = PlugDescriptor("outTransparency")
	pixelCenterX_ : PixelCenterXPlug = PlugDescriptor("pixelCenterX")
	pixelCenterY_ : PixelCenterYPlug = PlugDescriptor("pixelCenterY")
	pixelCenter_ : PixelCenterPlug = PlugDescriptor("pixelCenter")
	preFilter_ : PreFilterPlug = PlugDescriptor("preFilter")
	preFilterRadius_ : PreFilterRadiusPlug = PlugDescriptor("preFilterRadius")
	primitiveId_ : PrimitiveIdPlug = PlugDescriptor("primitiveId")
	ptexFilterBlur_ : PtexFilterBlurPlug = PlugDescriptor("ptexFilterBlur")
	ptexFilterInterpolateLevels_ : PtexFilterInterpolateLevelsPlug = PlugDescriptor("ptexFilterInterpolateLevels")
	ptexFilterSharpness_ : PtexFilterSharpnessPlug = PlugDescriptor("ptexFilterSharpness")
	ptexFilterType_ : PtexFilterTypePlug = PlugDescriptor("ptexFilterType")
	ptexFilterWidth_ : PtexFilterWidthPlug = PlugDescriptor("ptexFilterWidth")
	rayDepth_ : RayDepthPlug = PlugDescriptor("rayDepth")
	repeatU_ : RepeatUPlug = PlugDescriptor("repeatU")
	repeatV_ : RepeatVPlug = PlugDescriptor("repeatV")
	repeatUV_ : RepeatUVPlug = PlugDescriptor("repeatUV")
	rotateFrame_ : RotateFramePlug = PlugDescriptor("rotateFrame")
	rotateUV_ : RotateUVPlug = PlugDescriptor("rotateUV")
	stagger_ : StaggerPlug = PlugDescriptor("stagger")
	startCycleExtension_ : StartCycleExtensionPlug = PlugDescriptor("startCycleExtension")
	translateFrameU_ : TranslateFrameUPlug = PlugDescriptor("translateFrameU")
	translateFrameV_ : TranslateFrameVPlug = PlugDescriptor("translateFrameV")
	translateFrame_ : TranslateFramePlug = PlugDescriptor("translateFrame")
	useCache_ : UseCachePlug = PlugDescriptor("useCache")
	useFrameExtension_ : UseFrameExtensionPlug = PlugDescriptor("useFrameExtension")
	useHardwareTextureCycling_ : UseHardwareTextureCyclingPlug = PlugDescriptor("useHardwareTextureCycling")
	useMaximumRes_ : UseMaximumResPlug = PlugDescriptor("useMaximumRes")
	uvTileProxyDirty_ : UvTileProxyDirtyPlug = PlugDescriptor("uvTileProxyDirty")
	uvTileProxyGenerate_ : UvTileProxyGeneratePlug = PlugDescriptor("uvTileProxyGenerate")
	uvTileProxyQuality_ : UvTileProxyQualityPlug = PlugDescriptor("uvTileProxyQuality")
	uvTilingMode_ : UvTilingModePlug = PlugDescriptor("uvTilingMode")
	vertexCameraOneX_ : VertexCameraOneXPlug = PlugDescriptor("vertexCameraOneX")
	vertexCameraOneY_ : VertexCameraOneYPlug = PlugDescriptor("vertexCameraOneY")
	vertexCameraOneZ_ : VertexCameraOneZPlug = PlugDescriptor("vertexCameraOneZ")
	vertexCameraOne_ : VertexCameraOnePlug = PlugDescriptor("vertexCameraOne")
	vertexCameraThreeX_ : VertexCameraThreeXPlug = PlugDescriptor("vertexCameraThreeX")
	vertexCameraThreeY_ : VertexCameraThreeYPlug = PlugDescriptor("vertexCameraThreeY")
	vertexCameraThreeZ_ : VertexCameraThreeZPlug = PlugDescriptor("vertexCameraThreeZ")
	vertexCameraThree_ : VertexCameraThreePlug = PlugDescriptor("vertexCameraThree")
	vertexCameraTwoX_ : VertexCameraTwoXPlug = PlugDescriptor("vertexCameraTwoX")
	vertexCameraTwoY_ : VertexCameraTwoYPlug = PlugDescriptor("vertexCameraTwoY")
	vertexCameraTwoZ_ : VertexCameraTwoZPlug = PlugDescriptor("vertexCameraTwoZ")
	vertexCameraTwo_ : VertexCameraTwoPlug = PlugDescriptor("vertexCameraTwo")
	vertexUvOneU_ : VertexUvOneUPlug = PlugDescriptor("vertexUvOneU")
	vertexUvOneV_ : VertexUvOneVPlug = PlugDescriptor("vertexUvOneV")
	vertexUvOne_ : VertexUvOnePlug = PlugDescriptor("vertexUvOne")
	vertexUvThreeU_ : VertexUvThreeUPlug = PlugDescriptor("vertexUvThreeU")
	vertexUvThreeV_ : VertexUvThreeVPlug = PlugDescriptor("vertexUvThreeV")
	vertexUvThree_ : VertexUvThreePlug = PlugDescriptor("vertexUvThree")
	vertexUvTwoU_ : VertexUvTwoUPlug = PlugDescriptor("vertexUvTwoU")
	vertexUvTwoV_ : VertexUvTwoVPlug = PlugDescriptor("vertexUvTwoV")
	vertexUvTwo_ : VertexUvTwoPlug = PlugDescriptor("vertexUvTwo")
	viewNameStr_ : ViewNameStrPlug = PlugDescriptor("viewNameStr")
	viewNameUsed_ : ViewNameUsedPlug = PlugDescriptor("viewNameUsed")
	workingSpace_ : WorkingSpacePlug = PlugDescriptor("workingSpace")
	wrapU_ : WrapUPlug = PlugDescriptor("wrapU")
	wrapV_ : WrapVPlug = PlugDescriptor("wrapV")

	# node attributes

	typeName = "file"
	typeIdInt = 1381254740
	nodeLeafClassAttrs = ["baseExplicitUvTilePositionU", "baseExplicitUvTilePositionV", "baseExplicitUvTilePosition", "blurPixelation", "byCycleIncrement", "colorManagementConfigFileEnabled", "colorManagementConfigFilePath", "colorManagementEnabled", "colorProfile", "colorSpace", "computedFileTextureNamePattern", "coverageU", "coverageV", "coverage", "dirtyPixelRegion", "disableFileLoad", "doTransform", "endCycleExtension", "explicitUvTileName", "explicitUvTilePositionU", "explicitUvTilePositionV", "explicitUvTilePosition", "explicitUvTiles", "exposure", "fileHasAlpha", "fileTextureName", "fileTextureNamePattern", "filterType", "filterWidth", "forceSwatchGen", "frameExtension", "frameOffset", "hdrExposure", "hdrMapping", "ignoreColorSpaceFileRules", "infoBits", "mirrorU", "mirrorV", "noiseU", "noiseV", "noiseUV", "objectType", "offsetU", "offsetV", "offset", "outSizeX", "outSizeY", "outSize", "outTransparencyB", "outTransparencyG", "outTransparencyR", "outTransparency", "pixelCenterX", "pixelCenterY", "pixelCenter", "preFilter", "preFilterRadius", "primitiveId", "ptexFilterBlur", "ptexFilterInterpolateLevels", "ptexFilterSharpness", "ptexFilterType", "ptexFilterWidth", "rayDepth", "repeatU", "repeatV", "repeatUV", "rotateFrame", "rotateUV", "stagger", "startCycleExtension", "translateFrameU", "translateFrameV", "translateFrame", "useCache", "useFrameExtension", "useHardwareTextureCycling", "useMaximumRes", "uvTileProxyDirty", "uvTileProxyGenerate", "uvTileProxyQuality", "uvTilingMode", "vertexCameraOneX", "vertexCameraOneY", "vertexCameraOneZ", "vertexCameraOne", "vertexCameraThreeX", "vertexCameraThreeY", "vertexCameraThreeZ", "vertexCameraThree", "vertexCameraTwoX", "vertexCameraTwoY", "vertexCameraTwoZ", "vertexCameraTwo", "vertexUvOneU", "vertexUvOneV", "vertexUvOne", "vertexUvThreeU", "vertexUvThreeV", "vertexUvThree", "vertexUvTwoU", "vertexUvTwoV", "vertexUvTwo", "viewNameStr", "viewNameUsed", "workingSpace", "wrapU", "wrapV"]
	nodeLeafPlugs = ["baseExplicitUvTilePosition", "blurPixelation", "byCycleIncrement", "colorManagementConfigFileEnabled", "colorManagementConfigFilePath", "colorManagementEnabled", "colorProfile", "colorSpace", "computedFileTextureNamePattern", "coverage", "dirtyPixelRegion", "disableFileLoad", "doTransform", "endCycleExtension", "explicitUvTiles", "exposure", "fileHasAlpha", "fileTextureName", "fileTextureNamePattern", "filterType", "filterWidth", "forceSwatchGen", "frameExtension", "frameOffset", "hdrExposure", "hdrMapping", "ignoreColorSpaceFileRules", "infoBits", "mirrorU", "mirrorV", "noiseUV", "objectType", "offset", "outSize", "outTransparency", "pixelCenter", "preFilter", "preFilterRadius", "primitiveId", "ptexFilterBlur", "ptexFilterInterpolateLevels", "ptexFilterSharpness", "ptexFilterType", "ptexFilterWidth", "rayDepth", "repeatUV", "rotateFrame", "rotateUV", "stagger", "startCycleExtension", "translateFrame", "useCache", "useFrameExtension", "useHardwareTextureCycling", "useMaximumRes", "uvTileProxyDirty", "uvTileProxyGenerate", "uvTileProxyQuality", "uvTilingMode", "vertexCameraOne", "vertexCameraThree", "vertexCameraTwo", "vertexUvOne", "vertexUvThree", "vertexUvTwo", "viewNameStr", "viewNameUsed", "workingSpace", "wrapU", "wrapV"]
	pass

