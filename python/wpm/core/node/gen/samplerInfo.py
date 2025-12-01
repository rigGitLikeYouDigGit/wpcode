

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
class FacingRatioPlug(Plug):
	node : SamplerInfo = None
	pass
class FlippedNormalPlug(Plug):
	node : SamplerInfo = None
	pass
class MatrixEyeToWorldPlug(Plug):
	node : SamplerInfo = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : SamplerInfo = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : SamplerInfo = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : SamplerInfo = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	nx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ny_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	nz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : SamplerInfo = None
	pass
class PixelCenterXPlug(Plug):
	parent : PixelCenterPlug = PlugDescriptor("pixelCenter")
	node : SamplerInfo = None
	pass
class PixelCenterYPlug(Plug):
	parent : PixelCenterPlug = PlugDescriptor("pixelCenter")
	node : SamplerInfo = None
	pass
class PixelCenterPlug(Plug):
	pixelCenterX_ : PixelCenterXPlug = PlugDescriptor("pixelCenterX")
	pcx_ : PixelCenterXPlug = PlugDescriptor("pixelCenterX")
	pixelCenterY_ : PixelCenterYPlug = PlugDescriptor("pixelCenterY")
	pcy_ : PixelCenterYPlug = PlugDescriptor("pixelCenterY")
	node : SamplerInfo = None
	pass
class PointCameraXPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : SamplerInfo = None
	pass
class PointCameraYPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : SamplerInfo = None
	pass
class PointCameraZPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : SamplerInfo = None
	pass
class PointCameraPlug(Plug):
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	px_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	py_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pz_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	node : SamplerInfo = None
	pass
class PointObjXPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : SamplerInfo = None
	pass
class PointObjYPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : SamplerInfo = None
	pass
class PointObjZPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : SamplerInfo = None
	pass
class PointObjPlug(Plug):
	pointObjX_ : PointObjXPlug = PlugDescriptor("pointObjX")
	pox_ : PointObjXPlug = PlugDescriptor("pointObjX")
	pointObjY_ : PointObjYPlug = PlugDescriptor("pointObjY")
	poy_ : PointObjYPlug = PlugDescriptor("pointObjY")
	pointObjZ_ : PointObjZPlug = PlugDescriptor("pointObjZ")
	poz_ : PointObjZPlug = PlugDescriptor("pointObjZ")
	node : SamplerInfo = None
	pass
class PointWorldXPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : SamplerInfo = None
	pass
class PointWorldYPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : SamplerInfo = None
	pass
class PointWorldZPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : SamplerInfo = None
	pass
class PointWorldPlug(Plug):
	pointWorldX_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pwx_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pointWorldY_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pwy_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pointWorldZ_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	pwz_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	node : SamplerInfo = None
	pass
class RayDirectionXPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : SamplerInfo = None
	pass
class RayDirectionYPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : SamplerInfo = None
	pass
class RayDirectionZPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : SamplerInfo = None
	pass
class RayDirectionPlug(Plug):
	rayDirectionX_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rx_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rayDirectionY_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	ry_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rayDirectionZ_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	rz_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	node : SamplerInfo = None
	pass
class TangentUxPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : SamplerInfo = None
	pass
class TangentUyPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : SamplerInfo = None
	pass
class TangentUzPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : SamplerInfo = None
	pass
class TangentUCameraPlug(Plug):
	tangentUx_ : TangentUxPlug = PlugDescriptor("tangentUx")
	tux_ : TangentUxPlug = PlugDescriptor("tangentUx")
	tangentUy_ : TangentUyPlug = PlugDescriptor("tangentUy")
	tuy_ : TangentUyPlug = PlugDescriptor("tangentUy")
	tangentUz_ : TangentUzPlug = PlugDescriptor("tangentUz")
	tuz_ : TangentUzPlug = PlugDescriptor("tangentUz")
	node : SamplerInfo = None
	pass
class TangentVxPlug(Plug):
	parent : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	node : SamplerInfo = None
	pass
class TangentVyPlug(Plug):
	parent : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	node : SamplerInfo = None
	pass
class TangentVzPlug(Plug):
	parent : TangentVCameraPlug = PlugDescriptor("tangentVCamera")
	node : SamplerInfo = None
	pass
class TangentVCameraPlug(Plug):
	tangentVx_ : TangentVxPlug = PlugDescriptor("tangentVx")
	tvx_ : TangentVxPlug = PlugDescriptor("tangentVx")
	tangentVy_ : TangentVyPlug = PlugDescriptor("tangentVy")
	tvy_ : TangentVyPlug = PlugDescriptor("tangentVy")
	tangentVz_ : TangentVzPlug = PlugDescriptor("tangentVz")
	tvz_ : TangentVzPlug = PlugDescriptor("tangentVz")
	node : SamplerInfo = None
	pass
class UCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : SamplerInfo = None
	pass
class VCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : SamplerInfo = None
	pass
class UvCoordPlug(Plug):
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	u_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	v_ : VCoordPlug = PlugDescriptor("vCoord")
	node : SamplerInfo = None
	pass
# endregion


# define node class
class SamplerInfo(ShadingDependNode):
	facingRatio_ : FacingRatioPlug = PlugDescriptor("facingRatio")
	flippedNormal_ : FlippedNormalPlug = PlugDescriptor("flippedNormal")
	matrixEyeToWorld_ : MatrixEyeToWorldPlug = PlugDescriptor("matrixEyeToWorld")
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	normalCamera_ : NormalCameraPlug = PlugDescriptor("normalCamera")
	pixelCenterX_ : PixelCenterXPlug = PlugDescriptor("pixelCenterX")
	pixelCenterY_ : PixelCenterYPlug = PlugDescriptor("pixelCenterY")
	pixelCenter_ : PixelCenterPlug = PlugDescriptor("pixelCenter")
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pointCamera_ : PointCameraPlug = PlugDescriptor("pointCamera")
	pointObjX_ : PointObjXPlug = PlugDescriptor("pointObjX")
	pointObjY_ : PointObjYPlug = PlugDescriptor("pointObjY")
	pointObjZ_ : PointObjZPlug = PlugDescriptor("pointObjZ")
	pointObj_ : PointObjPlug = PlugDescriptor("pointObj")
	pointWorldX_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pointWorldY_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pointWorldZ_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	pointWorld_ : PointWorldPlug = PlugDescriptor("pointWorld")
	rayDirectionX_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rayDirectionY_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rayDirectionZ_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	rayDirection_ : RayDirectionPlug = PlugDescriptor("rayDirection")
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

	# node attributes

	typeName = "samplerInfo"
	apiTypeInt = 478
	apiTypeStr = "kSamplerInfo"
	typeIdInt = 1381189966
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["facingRatio", "flippedNormal", "matrixEyeToWorld", "normalCameraX", "normalCameraY", "normalCameraZ", "normalCamera", "pixelCenterX", "pixelCenterY", "pixelCenter", "pointCameraX", "pointCameraY", "pointCameraZ", "pointCamera", "pointObjX", "pointObjY", "pointObjZ", "pointObj", "pointWorldX", "pointWorldY", "pointWorldZ", "pointWorld", "rayDirectionX", "rayDirectionY", "rayDirectionZ", "rayDirection", "tangentUx", "tangentUy", "tangentUz", "tangentUCamera", "tangentVx", "tangentVy", "tangentVz", "tangentVCamera", "uCoord", "vCoord", "uvCoord"]
	nodeLeafPlugs = ["facingRatio", "flippedNormal", "matrixEyeToWorld", "normalCamera", "pixelCenter", "pointCamera", "pointObj", "pointWorld", "rayDirection", "tangentUCamera", "tangentVCamera", "uvCoord"]
	pass

