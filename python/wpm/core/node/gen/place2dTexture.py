

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ShadingDependNode = Catalogue.ShadingDependNode
else:
	from .. import retriever
	ShadingDependNode = retriever.getNodeCls("ShadingDependNode")
	assert ShadingDependNode

# add node doc



# region plug type defs
class CoverageUPlug(Plug):
	parent : CoveragePlug = PlugDescriptor("coverage")
	node : Place2dTexture = None
	pass
class CoverageVPlug(Plug):
	parent : CoveragePlug = PlugDescriptor("coverage")
	node : Place2dTexture = None
	pass
class CoveragePlug(Plug):
	coverageU_ : CoverageUPlug = PlugDescriptor("coverageU")
	cu_ : CoverageUPlug = PlugDescriptor("coverageU")
	coverageV_ : CoverageVPlug = PlugDescriptor("coverageV")
	cv_ : CoverageVPlug = PlugDescriptor("coverageV")
	node : Place2dTexture = None
	pass
class DoTransformPlug(Plug):
	node : Place2dTexture = None
	pass
class FastPlug(Plug):
	node : Place2dTexture = None
	pass
class MirrorUPlug(Plug):
	node : Place2dTexture = None
	pass
class MirrorVPlug(Plug):
	node : Place2dTexture = None
	pass
class NoiseUPlug(Plug):
	parent : NoiseUVPlug = PlugDescriptor("noiseUV")
	node : Place2dTexture = None
	pass
class NoiseVPlug(Plug):
	parent : NoiseUVPlug = PlugDescriptor("noiseUV")
	node : Place2dTexture = None
	pass
class NoiseUVPlug(Plug):
	noiseU_ : NoiseUPlug = PlugDescriptor("noiseU")
	nu_ : NoiseUPlug = PlugDescriptor("noiseU")
	noiseV_ : NoiseVPlug = PlugDescriptor("noiseV")
	nv_ : NoiseVPlug = PlugDescriptor("noiseV")
	node : Place2dTexture = None
	pass
class OffsetUPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : Place2dTexture = None
	pass
class OffsetVPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : Place2dTexture = None
	pass
class OffsetPlug(Plug):
	offsetU_ : OffsetUPlug = PlugDescriptor("offsetU")
	ofu_ : OffsetUPlug = PlugDescriptor("offsetU")
	offsetV_ : OffsetVPlug = PlugDescriptor("offsetV")
	ofv_ : OffsetVPlug = PlugDescriptor("offsetV")
	node : Place2dTexture = None
	pass
class OutUPlug(Plug):
	parent : OutUVPlug = PlugDescriptor("outUV")
	node : Place2dTexture = None
	pass
class OutVPlug(Plug):
	parent : OutUVPlug = PlugDescriptor("outUV")
	node : Place2dTexture = None
	pass
class OutUVPlug(Plug):
	outU_ : OutUPlug = PlugDescriptor("outU")
	ou_ : OutUPlug = PlugDescriptor("outU")
	outV_ : OutVPlug = PlugDescriptor("outV")
	ov_ : OutVPlug = PlugDescriptor("outV")
	node : Place2dTexture = None
	pass
class OutUvFilterSizeXPlug(Plug):
	parent : OutUvFilterSizePlug = PlugDescriptor("outUvFilterSize")
	node : Place2dTexture = None
	pass
class OutUvFilterSizeYPlug(Plug):
	parent : OutUvFilterSizePlug = PlugDescriptor("outUvFilterSize")
	node : Place2dTexture = None
	pass
class OutUvFilterSizePlug(Plug):
	outUvFilterSizeX_ : OutUvFilterSizeXPlug = PlugDescriptor("outUvFilterSizeX")
	ofsx_ : OutUvFilterSizeXPlug = PlugDescriptor("outUvFilterSizeX")
	outUvFilterSizeY_ : OutUvFilterSizeYPlug = PlugDescriptor("outUvFilterSizeY")
	ofsy_ : OutUvFilterSizeYPlug = PlugDescriptor("outUvFilterSizeY")
	node : Place2dTexture = None
	pass
class RepeatUPlug(Plug):
	parent : RepeatUVPlug = PlugDescriptor("repeatUV")
	node : Place2dTexture = None
	pass
class RepeatVPlug(Plug):
	parent : RepeatUVPlug = PlugDescriptor("repeatUV")
	node : Place2dTexture = None
	pass
class RepeatUVPlug(Plug):
	repeatU_ : RepeatUPlug = PlugDescriptor("repeatU")
	reu_ : RepeatUPlug = PlugDescriptor("repeatU")
	repeatV_ : RepeatVPlug = PlugDescriptor("repeatV")
	rev_ : RepeatVPlug = PlugDescriptor("repeatV")
	node : Place2dTexture = None
	pass
class RotateFramePlug(Plug):
	node : Place2dTexture = None
	pass
class RotateUVPlug(Plug):
	node : Place2dTexture = None
	pass
class StaggerPlug(Plug):
	node : Place2dTexture = None
	pass
class TranslateFrameUPlug(Plug):
	parent : TranslateFramePlug = PlugDescriptor("translateFrame")
	node : Place2dTexture = None
	pass
class TranslateFrameVPlug(Plug):
	parent : TranslateFramePlug = PlugDescriptor("translateFrame")
	node : Place2dTexture = None
	pass
class TranslateFramePlug(Plug):
	translateFrameU_ : TranslateFrameUPlug = PlugDescriptor("translateFrameU")
	tfu_ : TranslateFrameUPlug = PlugDescriptor("translateFrameU")
	translateFrameV_ : TranslateFrameVPlug = PlugDescriptor("translateFrameV")
	tfv_ : TranslateFrameVPlug = PlugDescriptor("translateFrameV")
	node : Place2dTexture = None
	pass
class UCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : Place2dTexture = None
	pass
class VCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : Place2dTexture = None
	pass
class UvCoordPlug(Plug):
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	u_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	v_ : VCoordPlug = PlugDescriptor("vCoord")
	node : Place2dTexture = None
	pass
class UvFilterSizeXPlug(Plug):
	parent : UvFilterSizePlug = PlugDescriptor("uvFilterSize")
	node : Place2dTexture = None
	pass
class UvFilterSizeYPlug(Plug):
	parent : UvFilterSizePlug = PlugDescriptor("uvFilterSize")
	node : Place2dTexture = None
	pass
class UvFilterSizePlug(Plug):
	uvFilterSizeX_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	fsx_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	uvFilterSizeY_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	fsy_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	node : Place2dTexture = None
	pass
class VertexCameraOneXPlug(Plug):
	parent : VertexCameraOnePlug = PlugDescriptor("vertexCameraOne")
	node : Place2dTexture = None
	pass
class VertexCameraOneYPlug(Plug):
	parent : VertexCameraOnePlug = PlugDescriptor("vertexCameraOne")
	node : Place2dTexture = None
	pass
class VertexCameraOneZPlug(Plug):
	parent : VertexCameraOnePlug = PlugDescriptor("vertexCameraOne")
	node : Place2dTexture = None
	pass
class VertexCameraOnePlug(Plug):
	vertexCameraOneX_ : VertexCameraOneXPlug = PlugDescriptor("vertexCameraOneX")
	c1x_ : VertexCameraOneXPlug = PlugDescriptor("vertexCameraOneX")
	vertexCameraOneY_ : VertexCameraOneYPlug = PlugDescriptor("vertexCameraOneY")
	c1y_ : VertexCameraOneYPlug = PlugDescriptor("vertexCameraOneY")
	vertexCameraOneZ_ : VertexCameraOneZPlug = PlugDescriptor("vertexCameraOneZ")
	c1z_ : VertexCameraOneZPlug = PlugDescriptor("vertexCameraOneZ")
	node : Place2dTexture = None
	pass
class VertexUvOneUPlug(Plug):
	parent : VertexUvOnePlug = PlugDescriptor("vertexUvOne")
	node : Place2dTexture = None
	pass
class VertexUvOneVPlug(Plug):
	parent : VertexUvOnePlug = PlugDescriptor("vertexUvOne")
	node : Place2dTexture = None
	pass
class VertexUvOnePlug(Plug):
	vertexUvOneU_ : VertexUvOneUPlug = PlugDescriptor("vertexUvOneU")
	t1u_ : VertexUvOneUPlug = PlugDescriptor("vertexUvOneU")
	vertexUvOneV_ : VertexUvOneVPlug = PlugDescriptor("vertexUvOneV")
	t1v_ : VertexUvOneVPlug = PlugDescriptor("vertexUvOneV")
	node : Place2dTexture = None
	pass
class VertexUvThreeUPlug(Plug):
	parent : VertexUvThreePlug = PlugDescriptor("vertexUvThree")
	node : Place2dTexture = None
	pass
class VertexUvThreeVPlug(Plug):
	parent : VertexUvThreePlug = PlugDescriptor("vertexUvThree")
	node : Place2dTexture = None
	pass
class VertexUvThreePlug(Plug):
	vertexUvThreeU_ : VertexUvThreeUPlug = PlugDescriptor("vertexUvThreeU")
	t3u_ : VertexUvThreeUPlug = PlugDescriptor("vertexUvThreeU")
	vertexUvThreeV_ : VertexUvThreeVPlug = PlugDescriptor("vertexUvThreeV")
	t3v_ : VertexUvThreeVPlug = PlugDescriptor("vertexUvThreeV")
	node : Place2dTexture = None
	pass
class VertexUvTwoUPlug(Plug):
	parent : VertexUvTwoPlug = PlugDescriptor("vertexUvTwo")
	node : Place2dTexture = None
	pass
class VertexUvTwoVPlug(Plug):
	parent : VertexUvTwoPlug = PlugDescriptor("vertexUvTwo")
	node : Place2dTexture = None
	pass
class VertexUvTwoPlug(Plug):
	vertexUvTwoU_ : VertexUvTwoUPlug = PlugDescriptor("vertexUvTwoU")
	t2u_ : VertexUvTwoUPlug = PlugDescriptor("vertexUvTwoU")
	vertexUvTwoV_ : VertexUvTwoVPlug = PlugDescriptor("vertexUvTwoV")
	t2v_ : VertexUvTwoVPlug = PlugDescriptor("vertexUvTwoV")
	node : Place2dTexture = None
	pass
class WrapUPlug(Plug):
	node : Place2dTexture = None
	pass
class WrapVPlug(Plug):
	node : Place2dTexture = None
	pass
# endregion


# define node class
class Place2dTexture(ShadingDependNode):
	coverageU_ : CoverageUPlug = PlugDescriptor("coverageU")
	coverageV_ : CoverageVPlug = PlugDescriptor("coverageV")
	coverage_ : CoveragePlug = PlugDescriptor("coverage")
	doTransform_ : DoTransformPlug = PlugDescriptor("doTransform")
	fast_ : FastPlug = PlugDescriptor("fast")
	mirrorU_ : MirrorUPlug = PlugDescriptor("mirrorU")
	mirrorV_ : MirrorVPlug = PlugDescriptor("mirrorV")
	noiseU_ : NoiseUPlug = PlugDescriptor("noiseU")
	noiseV_ : NoiseVPlug = PlugDescriptor("noiseV")
	noiseUV_ : NoiseUVPlug = PlugDescriptor("noiseUV")
	offsetU_ : OffsetUPlug = PlugDescriptor("offsetU")
	offsetV_ : OffsetVPlug = PlugDescriptor("offsetV")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	outU_ : OutUPlug = PlugDescriptor("outU")
	outV_ : OutVPlug = PlugDescriptor("outV")
	outUV_ : OutUVPlug = PlugDescriptor("outUV")
	outUvFilterSizeX_ : OutUvFilterSizeXPlug = PlugDescriptor("outUvFilterSizeX")
	outUvFilterSizeY_ : OutUvFilterSizeYPlug = PlugDescriptor("outUvFilterSizeY")
	outUvFilterSize_ : OutUvFilterSizePlug = PlugDescriptor("outUvFilterSize")
	repeatU_ : RepeatUPlug = PlugDescriptor("repeatU")
	repeatV_ : RepeatVPlug = PlugDescriptor("repeatV")
	repeatUV_ : RepeatUVPlug = PlugDescriptor("repeatUV")
	rotateFrame_ : RotateFramePlug = PlugDescriptor("rotateFrame")
	rotateUV_ : RotateUVPlug = PlugDescriptor("rotateUV")
	stagger_ : StaggerPlug = PlugDescriptor("stagger")
	translateFrameU_ : TranslateFrameUPlug = PlugDescriptor("translateFrameU")
	translateFrameV_ : TranslateFrameVPlug = PlugDescriptor("translateFrameV")
	translateFrame_ : TranslateFramePlug = PlugDescriptor("translateFrame")
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	uvCoord_ : UvCoordPlug = PlugDescriptor("uvCoord")
	uvFilterSizeX_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	uvFilterSizeY_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	uvFilterSize_ : UvFilterSizePlug = PlugDescriptor("uvFilterSize")
	vertexCameraOneX_ : VertexCameraOneXPlug = PlugDescriptor("vertexCameraOneX")
	vertexCameraOneY_ : VertexCameraOneYPlug = PlugDescriptor("vertexCameraOneY")
	vertexCameraOneZ_ : VertexCameraOneZPlug = PlugDescriptor("vertexCameraOneZ")
	vertexCameraOne_ : VertexCameraOnePlug = PlugDescriptor("vertexCameraOne")
	vertexUvOneU_ : VertexUvOneUPlug = PlugDescriptor("vertexUvOneU")
	vertexUvOneV_ : VertexUvOneVPlug = PlugDescriptor("vertexUvOneV")
	vertexUvOne_ : VertexUvOnePlug = PlugDescriptor("vertexUvOne")
	vertexUvThreeU_ : VertexUvThreeUPlug = PlugDescriptor("vertexUvThreeU")
	vertexUvThreeV_ : VertexUvThreeVPlug = PlugDescriptor("vertexUvThreeV")
	vertexUvThree_ : VertexUvThreePlug = PlugDescriptor("vertexUvThree")
	vertexUvTwoU_ : VertexUvTwoUPlug = PlugDescriptor("vertexUvTwoU")
	vertexUvTwoV_ : VertexUvTwoVPlug = PlugDescriptor("vertexUvTwoV")
	vertexUvTwo_ : VertexUvTwoPlug = PlugDescriptor("vertexUvTwo")
	wrapU_ : WrapUPlug = PlugDescriptor("wrapU")
	wrapV_ : WrapVPlug = PlugDescriptor("wrapV")

	# node attributes

	typeName = "place2dTexture"
	apiTypeInt = 457
	apiTypeStr = "kPlace2dTexture"
	typeIdInt = 1380994098
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["coverageU", "coverageV", "coverage", "doTransform", "fast", "mirrorU", "mirrorV", "noiseU", "noiseV", "noiseUV", "offsetU", "offsetV", "offset", "outU", "outV", "outUV", "outUvFilterSizeX", "outUvFilterSizeY", "outUvFilterSize", "repeatU", "repeatV", "repeatUV", "rotateFrame", "rotateUV", "stagger", "translateFrameU", "translateFrameV", "translateFrame", "uCoord", "vCoord", "uvCoord", "uvFilterSizeX", "uvFilterSizeY", "uvFilterSize", "vertexCameraOneX", "vertexCameraOneY", "vertexCameraOneZ", "vertexCameraOne", "vertexUvOneU", "vertexUvOneV", "vertexUvOne", "vertexUvThreeU", "vertexUvThreeV", "vertexUvThree", "vertexUvTwoU", "vertexUvTwoV", "vertexUvTwo", "wrapU", "wrapV"]
	nodeLeafPlugs = ["coverage", "doTransform", "fast", "mirrorU", "mirrorV", "noiseUV", "offset", "outUV", "outUvFilterSize", "repeatUV", "rotateFrame", "rotateUV", "stagger", "translateFrame", "uvCoord", "uvFilterSize", "vertexCameraOne", "vertexUvOne", "vertexUvThree", "vertexUvTwo", "wrapU", "wrapV"]
	pass

