

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Texture3d = retriever.getNodeCls("Texture3d")
assert Texture3d
if T.TYPE_CHECKING:
	from .. import Texture3d

# add node doc



# region plug type defs
class AmplitudeXPlug(Plug):
	node : Projection = None
	pass
class AmplitudeYPlug(Plug):
	node : Projection = None
	pass
class AngWtsPlug(Plug):
	node : Projection = None
	pass
class CamAngXPlug(Plug):
	parent : CamAgPlug = PlugDescriptor("camAg")
	node : Projection = None
	pass
class CamAngYPlug(Plug):
	parent : CamAgPlug = PlugDescriptor("camAg")
	node : Projection = None
	pass
class CamAngZPlug(Plug):
	parent : CamAgPlug = PlugDescriptor("camAg")
	node : Projection = None
	pass
class CamAgPlug(Plug):
	camAngX_ : CamAngXPlug = PlugDescriptor("camAngX")
	cax_ : CamAngXPlug = PlugDescriptor("camAngX")
	camAngY_ : CamAngYPlug = PlugDescriptor("camAngY")
	cay_ : CamAngYPlug = PlugDescriptor("camAngY")
	camAngZ_ : CamAngZPlug = PlugDescriptor("camAngZ")
	caz_ : CamAngZPlug = PlugDescriptor("camAngZ")
	node : Projection = None
	pass
class CamPsXPlug(Plug):
	parent : CamPosPlug = PlugDescriptor("camPos")
	node : Projection = None
	pass
class CamPsYPlug(Plug):
	parent : CamPosPlug = PlugDescriptor("camPos")
	node : Projection = None
	pass
class CamPsZPlug(Plug):
	parent : CamPosPlug = PlugDescriptor("camPos")
	node : Projection = None
	pass
class CamPosPlug(Plug):
	camPsX_ : CamPsXPlug = PlugDescriptor("camPsX")
	cpx_ : CamPsXPlug = PlugDescriptor("camPsX")
	camPsY_ : CamPsYPlug = PlugDescriptor("camPsY")
	cpy_ : CamPsYPlug = PlugDescriptor("camPsY")
	camPsZ_ : CamPsZPlug = PlugDescriptor("camPsZ")
	cpz_ : CamPsZPlug = PlugDescriptor("camPsZ")
	node : Projection = None
	pass
class DefaultTransparencyBPlug(Plug):
	parent : DefaultTransparencyPlug = PlugDescriptor("defaultTransparency")
	node : Projection = None
	pass
class DefaultTransparencyGPlug(Plug):
	parent : DefaultTransparencyPlug = PlugDescriptor("defaultTransparency")
	node : Projection = None
	pass
class DefaultTransparencyRPlug(Plug):
	parent : DefaultTransparencyPlug = PlugDescriptor("defaultTransparency")
	node : Projection = None
	pass
class DefaultTransparencyPlug(Plug):
	defaultTransparencyB_ : DefaultTransparencyBPlug = PlugDescriptor("defaultTransparencyB")
	dtb_ : DefaultTransparencyBPlug = PlugDescriptor("defaultTransparencyB")
	defaultTransparencyG_ : DefaultTransparencyGPlug = PlugDescriptor("defaultTransparencyG")
	dtg_ : DefaultTransparencyGPlug = PlugDescriptor("defaultTransparencyG")
	defaultTransparencyR_ : DefaultTransparencyRPlug = PlugDescriptor("defaultTransparencyR")
	dtr_ : DefaultTransparencyRPlug = PlugDescriptor("defaultTransparencyR")
	node : Projection = None
	pass
class DepWtsPlug(Plug):
	node : Projection = None
	pass
class DepthMaxPlug(Plug):
	parent : DepthPlug = PlugDescriptor("depth")
	node : Projection = None
	pass
class DepthMinPlug(Plug):
	parent : DepthPlug = PlugDescriptor("depth")
	node : Projection = None
	pass
class DepthPlug(Plug):
	depthMax_ : DepthMaxPlug = PlugDescriptor("depthMax")
	dmx_ : DepthMaxPlug = PlugDescriptor("depthMax")
	depthMin_ : DepthMinPlug = PlugDescriptor("depthMin")
	dmn_ : DepthMinPlug = PlugDescriptor("depthMin")
	node : Projection = None
	pass
class FitFillPlug(Plug):
	node : Projection = None
	pass
class FitTypePlug(Plug):
	node : Projection = None
	pass
class ImageBPlug(Plug):
	parent : ImagePlug = PlugDescriptor("image")
	node : Projection = None
	pass
class ImageGPlug(Plug):
	parent : ImagePlug = PlugDescriptor("image")
	node : Projection = None
	pass
class ImageRPlug(Plug):
	parent : ImagePlug = PlugDescriptor("image")
	node : Projection = None
	pass
class ImagePlug(Plug):
	imageB_ : ImageBPlug = PlugDescriptor("imageB")
	imb_ : ImageBPlug = PlugDescriptor("imageB")
	imageG_ : ImageGPlug = PlugDescriptor("imageG")
	img_ : ImageGPlug = PlugDescriptor("imageG")
	imageR_ : ImageRPlug = PlugDescriptor("imageR")
	imr_ : ImageRPlug = PlugDescriptor("imageR")
	node : Projection = None
	pass
class InfoBitsPlug(Plug):
	node : Projection = None
	pass
class LinkedCameraPlug(Plug):
	node : Projection = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Projection = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Projection = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Projection = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	nx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ny_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	nz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : Projection = None
	pass
class OutTransparencyBPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : Projection = None
	pass
class OutTransparencyGPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : Projection = None
	pass
class OutTransparencyRPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : Projection = None
	pass
class OutTransparencyPlug(Plug):
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	otb_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	otg_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	otr_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	node : Projection = None
	pass
class PassTrPlug(Plug):
	node : Projection = None
	pass
class ProjTypePlug(Plug):
	node : Projection = None
	pass
class RatioPlug(Plug):
	node : Projection = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Projection = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Projection = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Projection = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : Projection = None
	pass
class RefPointObjXPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Projection = None
	pass
class RefPointObjYPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Projection = None
	pass
class RefPointObjZPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Projection = None
	pass
class RefPointObjPlug(Plug):
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	rox_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	roy_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	roz_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	node : Projection = None
	pass
class RipplesXPlug(Plug):
	parent : RipplesPlug = PlugDescriptor("ripples")
	node : Projection = None
	pass
class RipplesYPlug(Plug):
	parent : RipplesPlug = PlugDescriptor("ripples")
	node : Projection = None
	pass
class RipplesZPlug(Plug):
	parent : RipplesPlug = PlugDescriptor("ripples")
	node : Projection = None
	pass
class RipplesPlug(Plug):
	ripplesX_ : RipplesXPlug = PlugDescriptor("ripplesX")
	rx_ : RipplesXPlug = PlugDescriptor("ripplesX")
	ripplesY_ : RipplesYPlug = PlugDescriptor("ripplesY")
	ry_ : RipplesYPlug = PlugDescriptor("ripplesY")
	ripplesZ_ : RipplesZPlug = PlugDescriptor("ripplesZ")
	rz_ : RipplesZPlug = PlugDescriptor("ripplesZ")
	node : Projection = None
	pass
class SrfNormalXPlug(Plug):
	parent : SrfNormalPlug = PlugDescriptor("srfNormal")
	node : Projection = None
	pass
class SrfNormalYPlug(Plug):
	parent : SrfNormalPlug = PlugDescriptor("srfNormal")
	node : Projection = None
	pass
class SrfNormalZPlug(Plug):
	parent : SrfNormalPlug = PlugDescriptor("srfNormal")
	node : Projection = None
	pass
class SrfNormalPlug(Plug):
	srfNormalX_ : SrfNormalXPlug = PlugDescriptor("srfNormalX")
	snx_ : SrfNormalXPlug = PlugDescriptor("srfNormalX")
	srfNormalY_ : SrfNormalYPlug = PlugDescriptor("srfNormalY")
	sny_ : SrfNormalYPlug = PlugDescriptor("srfNormalY")
	srfNormalZ_ : SrfNormalZPlug = PlugDescriptor("srfNormalZ")
	snz_ : SrfNormalZPlug = PlugDescriptor("srfNormalZ")
	node : Projection = None
	pass
class TangentUxPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : Projection = None
	pass
class TangentUyPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : Projection = None
	pass
class TangentUzPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : Projection = None
	pass
class TangentUCameraPlug(Plug):
	tangentUx_ : TangentUxPlug = PlugDescriptor("tangentUx")
	tux_ : TangentUxPlug = PlugDescriptor("tangentUx")
	tangentUy_ : TangentUyPlug = PlugDescriptor("tangentUy")
	tuy_ : TangentUyPlug = PlugDescriptor("tangentUy")
	tangentUz_ : TangentUzPlug = PlugDescriptor("tangentUz")
	tuz_ : TangentUzPlug = PlugDescriptor("tangentUz")
	node : Projection = None
	pass
class TangentVxPlug(Plug):
	parent : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	node : Projection = None
	pass
class TangentVyPlug(Plug):
	parent : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	node : Projection = None
	pass
class TangentVzPlug(Plug):
	parent : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	node : Projection = None
	pass
class TangentVCameraPlug(Plug):
	tangentVx_ : TangentVxPlug = PlugDescriptor("tangentVx")
	tvx_ : TangentVxPlug = PlugDescriptor("tangentVx")
	tangentVy_ : TangentVyPlug = PlugDescriptor("tangentVy")
	tvy_ : TangentVyPlug = PlugDescriptor("tangentVy")
	tangentVz_ : TangentVzPlug = PlugDescriptor("tangentVz")
	tvz_ : TangentVzPlug = PlugDescriptor("tangentVz")
	node : Projection = None
	pass
class TransparencyBPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : Projection = None
	pass
class TransparencyGPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : Projection = None
	pass
class TransparencyRPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : Projection = None
	pass
class TransparencyPlug(Plug):
	transparencyB_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	itb_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	transparencyG_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	itg_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	transparencyR_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	itr_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	node : Projection = None
	pass
class TransparencyGainBPlug(Plug):
	parent : TransparencyGainPlug = PlugDescriptor("transparencyGain")
	node : Projection = None
	pass
class TransparencyGainGPlug(Plug):
	parent : TransparencyGainPlug = PlugDescriptor("transparencyGain")
	node : Projection = None
	pass
class TransparencyGainRPlug(Plug):
	parent : TransparencyGainPlug = PlugDescriptor("transparencyGain")
	node : Projection = None
	pass
class TransparencyGainPlug(Plug):
	transparencyGainB_ : TransparencyGainBPlug = PlugDescriptor("transparencyGainB")
	tgb_ : TransparencyGainBPlug = PlugDescriptor("transparencyGainB")
	transparencyGainG_ : TransparencyGainGPlug = PlugDescriptor("transparencyGainG")
	tgg_ : TransparencyGainGPlug = PlugDescriptor("transparencyGainG")
	transparencyGainR_ : TransparencyGainRPlug = PlugDescriptor("transparencyGainR")
	tgr_ : TransparencyGainRPlug = PlugDescriptor("transparencyGainR")
	node : Projection = None
	pass
class TransparencyOffsetBPlug(Plug):
	parent : TransparencyOffsetPlug = PlugDescriptor("transparencyOffset")
	node : Projection = None
	pass
class TransparencyOffsetGPlug(Plug):
	parent : TransparencyOffsetPlug = PlugDescriptor("transparencyOffset")
	node : Projection = None
	pass
class TransparencyOffsetRPlug(Plug):
	parent : TransparencyOffsetPlug = PlugDescriptor("transparencyOffset")
	node : Projection = None
	pass
class TransparencyOffsetPlug(Plug):
	transparencyOffsetB_ : TransparencyOffsetBPlug = PlugDescriptor("transparencyOffsetB")
	tob_ : TransparencyOffsetBPlug = PlugDescriptor("transparencyOffsetB")
	transparencyOffsetG_ : TransparencyOffsetGPlug = PlugDescriptor("transparencyOffsetG")
	tog_ : TransparencyOffsetGPlug = PlugDescriptor("transparencyOffsetG")
	transparencyOffsetR_ : TransparencyOffsetRPlug = PlugDescriptor("transparencyOffsetR")
	tor_ : TransparencyOffsetRPlug = PlugDescriptor("transparencyOffsetR")
	node : Projection = None
	pass
class UAnglePlug(Plug):
	node : Projection = None
	pass
class UCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : Projection = None
	pass
class VCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : Projection = None
	pass
class UvCoordPlug(Plug):
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	u_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	v_ : VCoordPlug = PlugDescriptor("vCoord")
	node : Projection = None
	pass
class UvFilterSizeXPlug(Plug):
	parent : UvFilterSizePlug = PlugDescriptor("uvFilterSize")
	node : Projection = None
	pass
class UvFilterSizeYPlug(Plug):
	parent : UvFilterSizePlug = PlugDescriptor("uvFilterSize")
	node : Projection = None
	pass
class UvFilterSizePlug(Plug):
	uvFilterSizeX_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	ufx_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	uvFilterSizeY_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	ufy_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	node : Projection = None
	pass
class VAnglePlug(Plug):
	node : Projection = None
	pass
class VertexCameraOneXPlug(Plug):
	parent : VertexCameraOnePlug = PlugDescriptor("vertexCameraOne")
	node : Projection = None
	pass
class VertexCameraOneYPlug(Plug):
	parent : VertexCameraOnePlug = PlugDescriptor("vertexCameraOne")
	node : Projection = None
	pass
class VertexCameraOneZPlug(Plug):
	parent : VertexCameraOnePlug = PlugDescriptor("vertexCameraOne")
	node : Projection = None
	pass
class VertexCameraOnePlug(Plug):
	vertexCameraOneX_ : VertexCameraOneXPlug = PlugDescriptor("vertexCameraOneX")
	c1x_ : VertexCameraOneXPlug = PlugDescriptor("vertexCameraOneX")
	vertexCameraOneY_ : VertexCameraOneYPlug = PlugDescriptor("vertexCameraOneY")
	c1y_ : VertexCameraOneYPlug = PlugDescriptor("vertexCameraOneY")
	vertexCameraOneZ_ : VertexCameraOneZPlug = PlugDescriptor("vertexCameraOneZ")
	c1z_ : VertexCameraOneZPlug = PlugDescriptor("vertexCameraOneZ")
	node : Projection = None
	pass
class VertexCameraThreeXPlug(Plug):
	parent : VertexCameraThreePlug = PlugDescriptor("vertexCameraThree")
	node : Projection = None
	pass
class VertexCameraThreeYPlug(Plug):
	parent : VertexCameraThreePlug = PlugDescriptor("vertexCameraThree")
	node : Projection = None
	pass
class VertexCameraThreeZPlug(Plug):
	parent : VertexCameraThreePlug = PlugDescriptor("vertexCameraThree")
	node : Projection = None
	pass
class VertexCameraThreePlug(Plug):
	vertexCameraThreeX_ : VertexCameraThreeXPlug = PlugDescriptor("vertexCameraThreeX")
	c3x_ : VertexCameraThreeXPlug = PlugDescriptor("vertexCameraThreeX")
	vertexCameraThreeY_ : VertexCameraThreeYPlug = PlugDescriptor("vertexCameraThreeY")
	c3y_ : VertexCameraThreeYPlug = PlugDescriptor("vertexCameraThreeY")
	vertexCameraThreeZ_ : VertexCameraThreeZPlug = PlugDescriptor("vertexCameraThreeZ")
	c3z_ : VertexCameraThreeZPlug = PlugDescriptor("vertexCameraThreeZ")
	node : Projection = None
	pass
class VertexCameraTwoXPlug(Plug):
	parent : VertexCameraTwoPlug = PlugDescriptor("vertexCameraTwo")
	node : Projection = None
	pass
class VertexCameraTwoYPlug(Plug):
	parent : VertexCameraTwoPlug = PlugDescriptor("vertexCameraTwo")
	node : Projection = None
	pass
class VertexCameraTwoZPlug(Plug):
	parent : VertexCameraTwoPlug = PlugDescriptor("vertexCameraTwo")
	node : Projection = None
	pass
class VertexCameraTwoPlug(Plug):
	vertexCameraTwoX_ : VertexCameraTwoXPlug = PlugDescriptor("vertexCameraTwoX")
	c2x_ : VertexCameraTwoXPlug = PlugDescriptor("vertexCameraTwoX")
	vertexCameraTwoY_ : VertexCameraTwoYPlug = PlugDescriptor("vertexCameraTwoY")
	c2y_ : VertexCameraTwoYPlug = PlugDescriptor("vertexCameraTwoY")
	vertexCameraTwoZ_ : VertexCameraTwoZPlug = PlugDescriptor("vertexCameraTwoZ")
	c2z_ : VertexCameraTwoZPlug = PlugDescriptor("vertexCameraTwoZ")
	node : Projection = None
	pass
class VertexUvOneUPlug(Plug):
	parent : VertexUvOnePlug = PlugDescriptor("vertexUvOne")
	node : Projection = None
	pass
class VertexUvOneVPlug(Plug):
	parent : VertexUvOnePlug = PlugDescriptor("vertexUvOne")
	node : Projection = None
	pass
class VertexUvOnePlug(Plug):
	vertexUvOneU_ : VertexUvOneUPlug = PlugDescriptor("vertexUvOneU")
	t1u_ : VertexUvOneUPlug = PlugDescriptor("vertexUvOneU")
	vertexUvOneV_ : VertexUvOneVPlug = PlugDescriptor("vertexUvOneV")
	t1v_ : VertexUvOneVPlug = PlugDescriptor("vertexUvOneV")
	node : Projection = None
	pass
class VertexUvThreeUPlug(Plug):
	parent : VertexUvThreePlug = PlugDescriptor("vertexUvThree")
	node : Projection = None
	pass
class VertexUvThreeVPlug(Plug):
	parent : VertexUvThreePlug = PlugDescriptor("vertexUvThree")
	node : Projection = None
	pass
class VertexUvThreePlug(Plug):
	vertexUvThreeU_ : VertexUvThreeUPlug = PlugDescriptor("vertexUvThreeU")
	t3u_ : VertexUvThreeUPlug = PlugDescriptor("vertexUvThreeU")
	vertexUvThreeV_ : VertexUvThreeVPlug = PlugDescriptor("vertexUvThreeV")
	t3v_ : VertexUvThreeVPlug = PlugDescriptor("vertexUvThreeV")
	node : Projection = None
	pass
class VertexUvTwoUPlug(Plug):
	parent : VertexUvTwoPlug = PlugDescriptor("vertexUvTwo")
	node : Projection = None
	pass
class VertexUvTwoVPlug(Plug):
	parent : VertexUvTwoPlug = PlugDescriptor("vertexUvTwo")
	node : Projection = None
	pass
class VertexUvTwoPlug(Plug):
	vertexUvTwoU_ : VertexUvTwoUPlug = PlugDescriptor("vertexUvTwoU")
	t2u_ : VertexUvTwoUPlug = PlugDescriptor("vertexUvTwoU")
	vertexUvTwoV_ : VertexUvTwoVPlug = PlugDescriptor("vertexUvTwoV")
	t2v_ : VertexUvTwoVPlug = PlugDescriptor("vertexUvTwoV")
	node : Projection = None
	pass
class XPixelAnglePlug(Plug):
	node : Projection = None
	pass
# endregion


# define node class
class Projection(Texture3d):
	amplitudeX_ : AmplitudeXPlug = PlugDescriptor("amplitudeX")
	amplitudeY_ : AmplitudeYPlug = PlugDescriptor("amplitudeY")
	angWts_ : AngWtsPlug = PlugDescriptor("angWts")
	camAngX_ : CamAngXPlug = PlugDescriptor("camAngX")
	camAngY_ : CamAngYPlug = PlugDescriptor("camAngY")
	camAngZ_ : CamAngZPlug = PlugDescriptor("camAngZ")
	camAg_ : CamAgPlug = PlugDescriptor("camAg")
	camPsX_ : CamPsXPlug = PlugDescriptor("camPsX")
	camPsY_ : CamPsYPlug = PlugDescriptor("camPsY")
	camPsZ_ : CamPsZPlug = PlugDescriptor("camPsZ")
	camPos_ : CamPosPlug = PlugDescriptor("camPos")
	defaultTransparencyB_ : DefaultTransparencyBPlug = PlugDescriptor("defaultTransparencyB")
	defaultTransparencyG_ : DefaultTransparencyGPlug = PlugDescriptor("defaultTransparencyG")
	defaultTransparencyR_ : DefaultTransparencyRPlug = PlugDescriptor("defaultTransparencyR")
	defaultTransparency_ : DefaultTransparencyPlug = PlugDescriptor("defaultTransparency")
	depWts_ : DepWtsPlug = PlugDescriptor("depWts")
	depthMax_ : DepthMaxPlug = PlugDescriptor("depthMax")
	depthMin_ : DepthMinPlug = PlugDescriptor("depthMin")
	depth_ : DepthPlug = PlugDescriptor("depth")
	fitFill_ : FitFillPlug = PlugDescriptor("fitFill")
	fitType_ : FitTypePlug = PlugDescriptor("fitType")
	imageB_ : ImageBPlug = PlugDescriptor("imageB")
	imageG_ : ImageGPlug = PlugDescriptor("imageG")
	imageR_ : ImageRPlug = PlugDescriptor("imageR")
	image_ : ImagePlug = PlugDescriptor("image")
	infoBits_ : InfoBitsPlug = PlugDescriptor("infoBits")
	linkedCamera_ : LinkedCameraPlug = PlugDescriptor("linkedCamera")
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	normalCamera_ : NormalCameraPlug = PlugDescriptor("normalCamera")
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	outTransparency_ : OutTransparencyPlug = PlugDescriptor("outTransparency")
	passTr_ : PassTrPlug = PlugDescriptor("passTr")
	projType_ : ProjTypePlug = PlugDescriptor("projType")
	ratio_ : RatioPlug = PlugDescriptor("ratio")
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	refPointCamera_ : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	refPointObj_ : RefPointObjPlug = PlugDescriptor("refPointObj")
	ripplesX_ : RipplesXPlug = PlugDescriptor("ripplesX")
	ripplesY_ : RipplesYPlug = PlugDescriptor("ripplesY")
	ripplesZ_ : RipplesZPlug = PlugDescriptor("ripplesZ")
	ripples_ : RipplesPlug = PlugDescriptor("ripples")
	srfNormalX_ : SrfNormalXPlug = PlugDescriptor("srfNormalX")
	srfNormalY_ : SrfNormalYPlug = PlugDescriptor("srfNormalY")
	srfNormalZ_ : SrfNormalZPlug = PlugDescriptor("srfNormalZ")
	srfNormal_ : SrfNormalPlug = PlugDescriptor("srfNormal")
	tangentUx_ : TangentUxPlug = PlugDescriptor("tangentUx")
	tangentUy_ : TangentUyPlug = PlugDescriptor("tangentUy")
	tangentUz_ : TangentUzPlug = PlugDescriptor("tangentUz")
	tangentUCamera_ : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	tangentVx_ : TangentVxPlug = PlugDescriptor("tangentVx")
	tangentVy_ : TangentVyPlug = PlugDescriptor("tangentVy")
	tangentVz_ : TangentVzPlug = PlugDescriptor("tangentVz")
	tangentVCamera_ : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	transparencyB_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	transparencyG_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	transparencyR_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	transparency_ : TransparencyPlug = PlugDescriptor("transparency")
	transparencyGainB_ : TransparencyGainBPlug = PlugDescriptor("transparencyGainB")
	transparencyGainG_ : TransparencyGainGPlug = PlugDescriptor("transparencyGainG")
	transparencyGainR_ : TransparencyGainRPlug = PlugDescriptor("transparencyGainR")
	transparencyGain_ : TransparencyGainPlug = PlugDescriptor("transparencyGain")
	transparencyOffsetB_ : TransparencyOffsetBPlug = PlugDescriptor("transparencyOffsetB")
	transparencyOffsetG_ : TransparencyOffsetGPlug = PlugDescriptor("transparencyOffsetG")
	transparencyOffsetR_ : TransparencyOffsetRPlug = PlugDescriptor("transparencyOffsetR")
	transparencyOffset_ : TransparencyOffsetPlug = PlugDescriptor("transparencyOffset")
	uAngle_ : UAnglePlug = PlugDescriptor("uAngle")
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	uvCoord_ : UvCoordPlug = PlugDescriptor("uvCoord")
	uvFilterSizeX_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	uvFilterSizeY_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	uvFilterSize_ : UvFilterSizePlug = PlugDescriptor("uvFilterSize")
	vAngle_ : VAnglePlug = PlugDescriptor("vAngle")
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
	xPixelAngle_ : XPixelAnglePlug = PlugDescriptor("xPixelAngle")

	# node attributes

	typeName = "projection"
	apiTypeInt = 465
	apiTypeStr = "kProjection"
	typeIdInt = 1380995658
	MFnCls = om.MFnDependencyNode
	pass

