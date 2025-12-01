

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
	node : TextureToGeom = None
	pass
class ColorRangePlug(Plug):
	node : TextureToGeom = None
	pass
class FitTolerancePlug(Plug):
	node : TextureToGeom = None
	pass
class HardCornerDetectPlug(Plug):
	node : TextureToGeom = None
	pass
class HardCornerMaxLengthPlug(Plug):
	node : TextureToGeom = None
	pass
class ImageFilePlug(Plug):
	node : TextureToGeom = None
	pass
class InputMeshPlug(Plug):
	node : TextureToGeom = None
	pass
class InputMeshUVSetPlug(Plug):
	node : TextureToGeom = None
	pass
class MaxColorDiffPlug(Plug):
	node : TextureToGeom = None
	pass
class MaxPointsAddedPlug(Plug):
	node : TextureToGeom = None
	pass
class MeshQualityPlug(Plug):
	node : TextureToGeom = None
	pass
class MinSegmentSizePlug(Plug):
	node : TextureToGeom = None
	pass
class OutAlphaPlug(Plug):
	parent : OutColorDataPlug = PlugDescriptor("outColorData")
	node : TextureToGeom = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : TextureToGeom = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : TextureToGeom = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : TextureToGeom = None
	pass
class OutColorPlug(Plug):
	parent : OutColorDataPlug = PlugDescriptor("outColorData")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : TextureToGeom = None
	pass
class OutColorDataPlug(Plug):
	outAlpha_ : OutAlphaPlug = PlugDescriptor("outAlpha")
	oa_ : OutAlphaPlug = PlugDescriptor("outAlpha")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	oc_ : OutColorPlug = PlugDescriptor("outColor")
	node : TextureToGeom = None
	pass
class OutSegFacePlug(Plug):
	node : TextureToGeom = None
	pass
class OutputPlug(Plug):
	node : TextureToGeom = None
	pass
class PointsOnBoundaryPlug(Plug):
	node : TextureToGeom = None
	pass
class QuantizePlug(Plug):
	node : TextureToGeom = None
	pass
class QuantizeLevelsPlug(Plug):
	node : TextureToGeom = None
	pass
class SegGroupIdsPlug(Plug):
	node : TextureToGeom = None
	pass
class SegmentCountPlug(Plug):
	node : TextureToGeom = None
	pass
class ShaderScriptPlug(Plug):
	node : TextureToGeom = None
	pass
class SimplifyBoundaryPlug(Plug):
	node : TextureToGeom = None
	pass
class SimplifyThresholdPlug(Plug):
	node : TextureToGeom = None
	pass
class SmoothBoundaryPlug(Plug):
	node : TextureToGeom = None
	pass
class SmoothFactorPlug(Plug):
	node : TextureToGeom = None
	pass
class SpatialRadiusPlug(Plug):
	node : TextureToGeom = None
	pass
class SurfaceOffsetPlug(Plug):
	node : TextureToGeom = None
	pass
# endregion


# define node class
class TextureToGeom(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	colorRange_ : ColorRangePlug = PlugDescriptor("colorRange")
	fitTolerance_ : FitTolerancePlug = PlugDescriptor("fitTolerance")
	hardCornerDetect_ : HardCornerDetectPlug = PlugDescriptor("hardCornerDetect")
	hardCornerMaxLength_ : HardCornerMaxLengthPlug = PlugDescriptor("hardCornerMaxLength")
	imageFile_ : ImageFilePlug = PlugDescriptor("imageFile")
	inputMesh_ : InputMeshPlug = PlugDescriptor("inputMesh")
	inputMeshUVSet_ : InputMeshUVSetPlug = PlugDescriptor("inputMeshUVSet")
	maxColorDiff_ : MaxColorDiffPlug = PlugDescriptor("maxColorDiff")
	maxPointsAdded_ : MaxPointsAddedPlug = PlugDescriptor("maxPointsAdded")
	meshQuality_ : MeshQualityPlug = PlugDescriptor("meshQuality")
	minSegmentSize_ : MinSegmentSizePlug = PlugDescriptor("minSegmentSize")
	outAlpha_ : OutAlphaPlug = PlugDescriptor("outAlpha")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	outColorData_ : OutColorDataPlug = PlugDescriptor("outColorData")
	outSegFace_ : OutSegFacePlug = PlugDescriptor("outSegFace")
	output_ : OutputPlug = PlugDescriptor("output")
	pointsOnBoundary_ : PointsOnBoundaryPlug = PlugDescriptor("pointsOnBoundary")
	quantize_ : QuantizePlug = PlugDescriptor("quantize")
	quantizeLevels_ : QuantizeLevelsPlug = PlugDescriptor("quantizeLevels")
	segGroupIds_ : SegGroupIdsPlug = PlugDescriptor("segGroupIds")
	segmentCount_ : SegmentCountPlug = PlugDescriptor("segmentCount")
	shaderScript_ : ShaderScriptPlug = PlugDescriptor("shaderScript")
	simplifyBoundary_ : SimplifyBoundaryPlug = PlugDescriptor("simplifyBoundary")
	simplifyThreshold_ : SimplifyThresholdPlug = PlugDescriptor("simplifyThreshold")
	smoothBoundary_ : SmoothBoundaryPlug = PlugDescriptor("smoothBoundary")
	smoothFactor_ : SmoothFactorPlug = PlugDescriptor("smoothFactor")
	spatialRadius_ : SpatialRadiusPlug = PlugDescriptor("spatialRadius")
	surfaceOffset_ : SurfaceOffsetPlug = PlugDescriptor("surfaceOffset")

	# node attributes

	typeName = "textureToGeom"
	typeIdInt = 1414809423
	nodeLeafClassAttrs = ["binMembership", "colorRange", "fitTolerance", "hardCornerDetect", "hardCornerMaxLength", "imageFile", "inputMesh", "inputMeshUVSet", "maxColorDiff", "maxPointsAdded", "meshQuality", "minSegmentSize", "outAlpha", "outColorB", "outColorG", "outColorR", "outColor", "outColorData", "outSegFace", "output", "pointsOnBoundary", "quantize", "quantizeLevels", "segGroupIds", "segmentCount", "shaderScript", "simplifyBoundary", "simplifyThreshold", "smoothBoundary", "smoothFactor", "spatialRadius", "surfaceOffset"]
	nodeLeafPlugs = ["binMembership", "colorRange", "fitTolerance", "hardCornerDetect", "hardCornerMaxLength", "imageFile", "inputMesh", "inputMeshUVSet", "maxColorDiff", "maxPointsAdded", "meshQuality", "minSegmentSize", "outColorData", "outSegFace", "output", "pointsOnBoundary", "quantize", "quantizeLevels", "segGroupIds", "segmentCount", "shaderScript", "simplifyBoundary", "simplifyThreshold", "smoothBoundary", "smoothFactor", "spatialRadius", "surfaceOffset"]
	pass

