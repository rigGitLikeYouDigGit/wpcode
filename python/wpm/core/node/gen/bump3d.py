

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
class BinMembershipPlug(Plug):
	node : Bump3d = None
	pass
class BumpDepthPlug(Plug):
	node : Bump3d = None
	pass
class BumpFilterPlug(Plug):
	node : Bump3d = None
	pass
class BumpFilterOffsetPlug(Plug):
	node : Bump3d = None
	pass
class BumpValuePlug(Plug):
	node : Bump3d = None
	pass
class InfoBitsPlug(Plug):
	node : Bump3d = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Bump3d = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Bump3d = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Bump3d = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	nx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ny_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	nz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : Bump3d = None
	pass
class OutNormalXPlug(Plug):
	parent : OutNormalPlug = PlugDescriptor("outNormal")
	node : Bump3d = None
	pass
class OutNormalYPlug(Plug):
	parent : OutNormalPlug = PlugDescriptor("outNormal")
	node : Bump3d = None
	pass
class OutNormalZPlug(Plug):
	parent : OutNormalPlug = PlugDescriptor("outNormal")
	node : Bump3d = None
	pass
class OutNormalPlug(Plug):
	outNormalX_ : OutNormalXPlug = PlugDescriptor("outNormalX")
	ox_ : OutNormalXPlug = PlugDescriptor("outNormalX")
	outNormalY_ : OutNormalYPlug = PlugDescriptor("outNormalY")
	oy_ : OutNormalYPlug = PlugDescriptor("outNormalY")
	outNormalZ_ : OutNormalZPlug = PlugDescriptor("outNormalZ")
	oz_ : OutNormalZPlug = PlugDescriptor("outNormalZ")
	node : Bump3d = None
	pass
class PointCameraXPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : Bump3d = None
	pass
class PointCameraYPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : Bump3d = None
	pass
class PointCameraZPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : Bump3d = None
	pass
class PointCameraPlug(Plug):
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	px_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	py_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pz_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	node : Bump3d = None
	pass
class PointObjXPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : Bump3d = None
	pass
class PointObjYPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : Bump3d = None
	pass
class PointObjZPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : Bump3d = None
	pass
class PointObjPlug(Plug):
	pointObjX_ : PointObjXPlug = PlugDescriptor("pointObjX")
	pox_ : PointObjXPlug = PlugDescriptor("pointObjX")
	pointObjY_ : PointObjYPlug = PlugDescriptor("pointObjY")
	poy_ : PointObjYPlug = PlugDescriptor("pointObjY")
	pointObjZ_ : PointObjZPlug = PlugDescriptor("pointObjZ")
	poz_ : PointObjZPlug = PlugDescriptor("pointObjZ")
	node : Bump3d = None
	pass
class RayOriginXPlug(Plug):
	parent : RayOriginPlug = PlugDescriptor("rayOrigin")
	node : Bump3d = None
	pass
class RayOriginYPlug(Plug):
	parent : RayOriginPlug = PlugDescriptor("rayOrigin")
	node : Bump3d = None
	pass
class RayOriginZPlug(Plug):
	parent : RayOriginPlug = PlugDescriptor("rayOrigin")
	node : Bump3d = None
	pass
class RayOriginPlug(Plug):
	rayOriginX_ : RayOriginXPlug = PlugDescriptor("rayOriginX")
	rox_ : RayOriginXPlug = PlugDescriptor("rayOriginX")
	rayOriginY_ : RayOriginYPlug = PlugDescriptor("rayOriginY")
	roy_ : RayOriginYPlug = PlugDescriptor("rayOriginY")
	rayOriginZ_ : RayOriginZPlug = PlugDescriptor("rayOriginZ")
	roz_ : RayOriginZPlug = PlugDescriptor("rayOriginZ")
	node : Bump3d = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Bump3d = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Bump3d = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : Bump3d = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : Bump3d = None
	pass
class RefPointObjXPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Bump3d = None
	pass
class RefPointObjYPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Bump3d = None
	pass
class RefPointObjZPlug(Plug):
	parent : RefPointObjPlug = PlugDescriptor("refPointObj")
	node : Bump3d = None
	pass
class RefPointObjPlug(Plug):
	refPointObjX_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	rpox_ : RefPointObjXPlug = PlugDescriptor("refPointObjX")
	refPointObjY_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	rpoy_ : RefPointObjYPlug = PlugDescriptor("refPointObjY")
	refPointObjZ_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	rpoz_ : RefPointObjZPlug = PlugDescriptor("refPointObjZ")
	node : Bump3d = None
	pass
class TangentUxPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : Bump3d = None
	pass
class TangentUyPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : Bump3d = None
	pass
class TangentUzPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : Bump3d = None
	pass
class TangentUCameraPlug(Plug):
	tangentUx_ : TangentUxPlug = PlugDescriptor("tangentUx")
	tux_ : TangentUxPlug = PlugDescriptor("tangentUx")
	tangentUy_ : TangentUyPlug = PlugDescriptor("tangentUy")
	tuy_ : TangentUyPlug = PlugDescriptor("tangentUy")
	tangentUz_ : TangentUzPlug = PlugDescriptor("tangentUz")
	tuz_ : TangentUzPlug = PlugDescriptor("tangentUz")
	node : Bump3d = None
	pass
class TangentVxPlug(Plug):
	parent : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	node : Bump3d = None
	pass
class TangentVyPlug(Plug):
	parent : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	node : Bump3d = None
	pass
class TangentVzPlug(Plug):
	parent : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	node : Bump3d = None
	pass
class TangentVCameraPlug(Plug):
	tangentVx_ : TangentVxPlug = PlugDescriptor("tangentVx")
	tvx_ : TangentVxPlug = PlugDescriptor("tangentVx")
	tangentVy_ : TangentVyPlug = PlugDescriptor("tangentVy")
	tvy_ : TangentVyPlug = PlugDescriptor("tangentVy")
	tangentVz_ : TangentVzPlug = PlugDescriptor("tangentVz")
	tvz_ : TangentVzPlug = PlugDescriptor("tangentVz")
	node : Bump3d = None
	pass
class XPixelAnglePlug(Plug):
	node : Bump3d = None
	pass
# endregion


# define node class
class Bump3d(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	bumpDepth_ : BumpDepthPlug = PlugDescriptor("bumpDepth")
	bumpFilter_ : BumpFilterPlug = PlugDescriptor("bumpFilter")
	bumpFilterOffset_ : BumpFilterOffsetPlug = PlugDescriptor("bumpFilterOffset")
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
	xPixelAngle_ : XPixelAnglePlug = PlugDescriptor("xPixelAngle")

	# node attributes

	typeName = "bump3d"
	apiTypeInt = 33
	apiTypeStr = "kBump3d"
	typeIdInt = 1380078899
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "bumpDepth", "bumpFilter", "bumpFilterOffset", "bumpValue", "infoBits", "normalCameraX", "normalCameraY", "normalCameraZ", "normalCamera", "outNormalX", "outNormalY", "outNormalZ", "outNormal", "pointCameraX", "pointCameraY", "pointCameraZ", "pointCamera", "pointObjX", "pointObjY", "pointObjZ", "pointObj", "rayOriginX", "rayOriginY", "rayOriginZ", "rayOrigin", "refPointCameraX", "refPointCameraY", "refPointCameraZ", "refPointCamera", "refPointObjX", "refPointObjY", "refPointObjZ", "refPointObj", "tangentUx", "tangentUy", "tangentUz", "tangentUCamera", "tangentVx", "tangentVy", "tangentVz", "tangentVCamera", "xPixelAngle"]
	nodeLeafPlugs = ["binMembership", "bumpDepth", "bumpFilter", "bumpFilterOffset", "bumpValue", "infoBits", "normalCamera", "outNormal", "pointCamera", "pointObj", "rayOrigin", "refPointCamera", "refPointObj", "tangentUCamera", "tangentVCamera", "xPixelAngle"]
	pass

