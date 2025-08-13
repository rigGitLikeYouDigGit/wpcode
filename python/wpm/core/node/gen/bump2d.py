

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ShadingDependNode = retriever.getNodeCls("ShadingDependNode")
assert ShadingDependNode
if T.TYPE_CHECKING:
	from .. import ShadingDependNode

# add node doc



# region plug type defs
class AdjustEdgesPlug(Plug):
	node : Bump2d = None
	pass
class BumpDepthPlug(Plug):
	node : Bump2d = None
	pass
class BumpFilterPlug(Plug):
	node : Bump2d = None
	pass
class BumpFilterOffsetPlug(Plug):
	node : Bump2d = None
	pass
class BumpInterpPlug(Plug):
	node : Bump2d = None
	pass
class BumpValuePlug(Plug):
	node : Bump2d = None
	pass
class InfoBitsPlug(Plug):
	node : Bump2d = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Bump2d = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Bump2d = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Bump2d = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	nx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ny_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	nz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : Bump2d = None
	pass
class OutNormalXPlug(Plug):
	parent : OutNormalPlug = PlugDescriptor("outNormal")
	node : Bump2d = None
	pass
class OutNormalYPlug(Plug):
	parent : OutNormalPlug = PlugDescriptor("outNormal")
	node : Bump2d = None
	pass
class OutNormalZPlug(Plug):
	parent : OutNormalPlug = PlugDescriptor("outNormal")
	node : Bump2d = None
	pass
class OutNormalPlug(Plug):
	outNormalX_ : OutNormalXPlug = PlugDescriptor("outNormalX")
	ox_ : OutNormalXPlug = PlugDescriptor("outNormalX")
	outNormalY_ : OutNormalYPlug = PlugDescriptor("outNormalY")
	oy_ : OutNormalYPlug = PlugDescriptor("outNormalY")
	outNormalZ_ : OutNormalZPlug = PlugDescriptor("outNormalZ")
	oz_ : OutNormalZPlug = PlugDescriptor("outNormalZ")
	node : Bump2d = None
	pass
class PointCameraXPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : Bump2d = None
	pass
class PointCameraYPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : Bump2d = None
	pass
class PointCameraZPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : Bump2d = None
	pass
class PointCameraPlug(Plug):
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	px_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	py_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pz_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	node : Bump2d = None
	pass
class PointObjXPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : Bump2d = None
	pass
class PointObjYPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : Bump2d = None
	pass
class PointObjZPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : Bump2d = None
	pass
class PointObjPlug(Plug):
	pointObjX_ : PointObjXPlug = PlugDescriptor("pointObjX")
	pox_ : PointObjXPlug = PlugDescriptor("pointObjX")
	pointObjY_ : PointObjYPlug = PlugDescriptor("pointObjY")
	poy_ : PointObjYPlug = PlugDescriptor("pointObjY")
	pointObjZ_ : PointObjZPlug = PlugDescriptor("pointObjZ")
	poz_ : PointObjZPlug = PlugDescriptor("pointObjZ")
	node : Bump2d = None
	pass
class Provide3dInfoPlug(Plug):
	node : Bump2d = None
	pass
class RayOriginXPlug(Plug):
	parent : RayOriginPlug = PlugDescriptor("rayOrigin")
	node : Bump2d = None
	pass
class RayOriginYPlug(Plug):
	parent : RayOriginPlug = PlugDescriptor("rayOrigin")
	node : Bump2d = None
	pass
class RayOriginZPlug(Plug):
	parent : RayOriginPlug = PlugDescriptor("rayOrigin")
	node : Bump2d = None
	pass
class RayOriginPlug(Plug):
	rayOriginX_ : RayOriginXPlug = PlugDescriptor("rayOriginX")
	rox_ : RayOriginXPlug = PlugDescriptor("rayOriginX")
	rayOriginY_ : RayOriginYPlug = PlugDescriptor("rayOriginY")
	roy_ : RayOriginYPlug = PlugDescriptor("rayOriginY")
	rayOriginZ_ : RayOriginZPlug = PlugDescriptor("rayOriginZ")
	roz_ : RayOriginZPlug = PlugDescriptor("rayOriginZ")
	node : Bump2d = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Bump2d = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Bump2d = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Bump2d = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : Bump2d = None
	pass
class RefPointObjXPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Bump2d = None
	pass
class RefPointObjYPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Bump2d = None
	pass
class RefPointObjZPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Bump2d = None
	pass
class RefPointObjPlug(Plug):
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	rpox_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	rpoy_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	rpoz_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	node : Bump2d = None
	pass
class TangentUxPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : Bump2d = None
	pass
class TangentUyPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : Bump2d = None
	pass
class TangentUzPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : Bump2d = None
	pass
class TangentUCameraPlug(Plug):
	tangentUx_ : TangentUxPlug = PlugDescriptor("tangentUx")
	tux_ : TangentUxPlug = PlugDescriptor("tangentUx")
	tangentUy_ : TangentUyPlug = PlugDescriptor("tangentUy")
	tuy_ : TangentUyPlug = PlugDescriptor("tangentUy")
	tangentUz_ : TangentUzPlug = PlugDescriptor("tangentUz")
	tuz_ : TangentUzPlug = PlugDescriptor("tangentUz")
	node : Bump2d = None
	pass
class TangentVxPlug(Plug):
	parent : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	node : Bump2d = None
	pass
class TangentVyPlug(Plug):
	parent : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	node : Bump2d = None
	pass
class TangentVzPlug(Plug):
	parent : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	node : Bump2d = None
	pass
class TangentVCameraPlug(Plug):
	tangentVx_ : TangentVxPlug = PlugDescriptor("tangentVx")
	tvx_ : TangentVxPlug = PlugDescriptor("tangentVx")
	tangentVy_ : TangentVyPlug = PlugDescriptor("tangentVy")
	tvy_ : TangentVyPlug = PlugDescriptor("tangentVy")
	tangentVz_ : TangentVzPlug = PlugDescriptor("tangentVz")
	tvz_ : TangentVzPlug = PlugDescriptor("tangentVz")
	node : Bump2d = None
	pass
class UCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : Bump2d = None
	pass
class VCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : Bump2d = None
	pass
class UvCoordPlug(Plug):
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	u_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	v_ : VCoordPlug = PlugDescriptor("vCoord")
	node : Bump2d = None
	pass
class UvFilterSizeXPlug(Plug):
	parent : UvFilterSizePlug = PlugDescriptor("uvFilterSize")
	node : Bump2d = None
	pass
class UvFilterSizeYPlug(Plug):
	parent : UvFilterSizePlug = PlugDescriptor("uvFilterSize")
	node : Bump2d = None
	pass
class UvFilterSizePlug(Plug):
	uvFilterSizeX_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	fsx_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	uvFilterSizeY_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	fsy_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	node : Bump2d = None
	pass
class VertexCameraOneXPlug(Plug):
	parent : VertexCameraOnePlug = PlugDescriptor("vertexCameraOne")
	node : Bump2d = None
	pass
class VertexCameraOneYPlug(Plug):
	parent : VertexCameraOnePlug = PlugDescriptor("vertexCameraOne")
	node : Bump2d = None
	pass
class VertexCameraOneZPlug(Plug):
	parent : VertexCameraOnePlug = PlugDescriptor("vertexCameraOne")
	node : Bump2d = None
	pass
class VertexCameraOnePlug(Plug):
	vertexCameraOneX_ : VertexCameraOneXPlug = PlugDescriptor("vertexCameraOneX")
	c1x_ : VertexCameraOneXPlug = PlugDescriptor("vertexCameraOneX")
	vertexCameraOneY_ : VertexCameraOneYPlug = PlugDescriptor("vertexCameraOneY")
	c1y_ : VertexCameraOneYPlug = PlugDescriptor("vertexCameraOneY")
	vertexCameraOneZ_ : VertexCameraOneZPlug = PlugDescriptor("vertexCameraOneZ")
	c1z_ : VertexCameraOneZPlug = PlugDescriptor("vertexCameraOneZ")
	node : Bump2d = None
	pass
class VertexCameraTwoXPlug(Plug):
	parent : VertexCameraTwoPlug = PlugDescriptor("vertexCameraTwo")
	node : Bump2d = None
	pass
class VertexCameraTwoYPlug(Plug):
	parent : VertexCameraTwoPlug = PlugDescriptor("vertexCameraTwo")
	node : Bump2d = None
	pass
class VertexCameraTwoZPlug(Plug):
	parent : VertexCameraTwoPlug = PlugDescriptor("vertexCameraTwo")
	node : Bump2d = None
	pass
class VertexCameraTwoPlug(Plug):
	vertexCameraTwoX_ : VertexCameraTwoXPlug = PlugDescriptor("vertexCameraTwoX")
	c2x_ : VertexCameraTwoXPlug = PlugDescriptor("vertexCameraTwoX")
	vertexCameraTwoY_ : VertexCameraTwoYPlug = PlugDescriptor("vertexCameraTwoY")
	c2y_ : VertexCameraTwoYPlug = PlugDescriptor("vertexCameraTwoY")
	vertexCameraTwoZ_ : VertexCameraTwoZPlug = PlugDescriptor("vertexCameraTwoZ")
	c2z_ : VertexCameraTwoZPlug = PlugDescriptor("vertexCameraTwoZ")
	node : Bump2d = None
	pass
class VertexUvOneUPlug(Plug):
	parent : VertexUvOnePlug = PlugDescriptor("vertexUvOne")
	node : Bump2d = None
	pass
class VertexUvOneVPlug(Plug):
	parent : VertexUvOnePlug = PlugDescriptor("vertexUvOne")
	node : Bump2d = None
	pass
class VertexUvOnePlug(Plug):
	vertexUvOneU_ : VertexUvOneUPlug = PlugDescriptor("vertexUvOneU")
	t1u_ : VertexUvOneUPlug = PlugDescriptor("vertexUvOneU")
	vertexUvOneV_ : VertexUvOneVPlug = PlugDescriptor("vertexUvOneV")
	t1v_ : VertexUvOneVPlug = PlugDescriptor("vertexUvOneV")
	node : Bump2d = None
	pass
class VertexUvTwoUPlug(Plug):
	parent : VertexUvTwoPlug = PlugDescriptor("vertexUvTwo")
	node : Bump2d = None
	pass
class VertexUvTwoVPlug(Plug):
	parent : VertexUvTwoPlug = PlugDescriptor("vertexUvTwo")
	node : Bump2d = None
	pass
class VertexUvTwoPlug(Plug):
	vertexUvTwoU_ : VertexUvTwoUPlug = PlugDescriptor("vertexUvTwoU")
	t2u_ : VertexUvTwoUPlug = PlugDescriptor("vertexUvTwoU")
	vertexUvTwoV_ : VertexUvTwoVPlug = PlugDescriptor("vertexUvTwoV")
	t2v_ : VertexUvTwoVPlug = PlugDescriptor("vertexUvTwoV")
	node : Bump2d = None
	pass
class XPixelAnglePlug(Plug):
	node : Bump2d = None
	pass
# endregion


# define node class
class Bump2d(ShadingDependNode):
	adjustEdges_ : AdjustEdgesPlug = PlugDescriptor("adjustEdges")
	bumpDepth_ : BumpDepthPlug = PlugDescriptor("bumpDepth")
	bumpFilter_ : BumpFilterPlug = PlugDescriptor("bumpFilter")
	bumpFilterOffset_ : BumpFilterOffsetPlug = PlugDescriptor("bumpFilterOffset")
	bumpInterp_ : BumpInterpPlug = PlugDescriptor("bumpInterp")
	bumpValue_ : BumpValuePlug = PlugDescriptor("bumpValue")
	infoBits_ : InfoBitsPlug = PlugDescriptor("infoBits")
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	normalCamera_ : NormalCameraPlug = PlugDescriptor("normalCamera")
	outNormalX_ : OutNormalXPlug = PlugDescriptor("outNormalX")
	outNormalY_ : OutNormalYPlug = PlugDescriptor("outNormalY")
	outNormalZ_ : OutNormalZPlug = PlugDescriptor("outNormalZ")
	outNormal_ : OutNormalPlug = PlugDescriptor("outNormal")
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pointCamera_ : PointCameraPlug = PlugDescriptor("pointCamera")
	pointObjX_ : PointObjXPlug = PlugDescriptor("pointObjX")
	pointObjY_ : PointObjYPlug = PlugDescriptor("pointObjY")
	pointObjZ_ : PointObjZPlug = PlugDescriptor("pointObjZ")
	pointObj_ : PointObjPlug = PlugDescriptor("pointObj")
	provide3dInfo_ : Provide3dInfoPlug = PlugDescriptor("provide3dInfo")
	rayOriginX_ : RayOriginXPlug = PlugDescriptor("rayOriginX")
	rayOriginY_ : RayOriginYPlug = PlugDescriptor("rayOriginY")
	rayOriginZ_ : RayOriginZPlug = PlugDescriptor("rayOriginZ")
	rayOrigin_ : RayOriginPlug = PlugDescriptor("rayOrigin")
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	refPointCamera_ : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	refPointObj_ : RefPointObjPlug = PlugDescriptor("refPointObj")
	tangentUx_ : TangentUxPlug = PlugDescriptor("tangentUx")
	tangentUy_ : TangentUyPlug = PlugDescriptor("tangentUy")
	tangentUz_ : TangentUzPlug = PlugDescriptor("tangentUz")
	tangentUCamera_ : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	tangentVx_ : TangentVxPlug = PlugDescriptor("tangentVx")
	tangentVy_ : TangentVyPlug = PlugDescriptor("tangentVy")
	tangentVz_ : TangentVzPlug = PlugDescriptor("tangentVz")
	tangentVCamera_ : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
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
	vertexCameraTwoX_ : VertexCameraTwoXPlug = PlugDescriptor("vertexCameraTwoX")
	vertexCameraTwoY_ : VertexCameraTwoYPlug = PlugDescriptor("vertexCameraTwoY")
	vertexCameraTwoZ_ : VertexCameraTwoZPlug = PlugDescriptor("vertexCameraTwoZ")
	vertexCameraTwo_ : VertexCameraTwoPlug = PlugDescriptor("vertexCameraTwo")
	vertexUvOneU_ : VertexUvOneUPlug = PlugDescriptor("vertexUvOneU")
	vertexUvOneV_ : VertexUvOneVPlug = PlugDescriptor("vertexUvOneV")
	vertexUvOne_ : VertexUvOnePlug = PlugDescriptor("vertexUvOne")
	vertexUvTwoU_ : VertexUvTwoUPlug = PlugDescriptor("vertexUvTwoU")
	vertexUvTwoV_ : VertexUvTwoVPlug = PlugDescriptor("vertexUvTwoV")
	vertexUvTwo_ : VertexUvTwoPlug = PlugDescriptor("vertexUvTwo")
	xPixelAngle_ : XPixelAnglePlug = PlugDescriptor("xPixelAngle")

	# node attributes

	typeName = "bump2d"
	typeIdInt = 1380078925
	pass

