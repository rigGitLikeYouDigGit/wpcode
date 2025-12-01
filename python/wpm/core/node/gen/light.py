

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
class CenterOfIlluminationPlug(Plug):
	node : Light = None
	pass
class ColorBPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : Light = None
	pass
class ColorGPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : Light = None
	pass
class ColorRPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : Light = None
	pass
class ColorPlug(Plug):
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	cb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	cg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	cr_ : ColorRPlug = PlugDescriptor("colorR")
	node : Light = None
	pass
class InfoBitsPlug(Plug):
	node : Light = None
	pass
class IntensityPlug(Plug):
	node : Light = None
	pass
class LightAmbientPlug(Plug):
	parent : LightDataPlug = PlugDescriptor("lightData")
	node : Light = None
	pass
class LightBlindDataPlug(Plug):
	parent : LightDataPlug = PlugDescriptor("lightData")
	node : Light = None
	pass
class LightDiffusePlug(Plug):
	parent : LightDataPlug = PlugDescriptor("lightData")
	node : Light = None
	pass
class LightDirectionXPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : Light = None
	pass
class LightDirectionYPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : Light = None
	pass
class LightDirectionZPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : Light = None
	pass
class LightDirectionPlug(Plug):
	parent : LightDataPlug = PlugDescriptor("lightData")
	lightDirectionX_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	ldx_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	lightDirectionY_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	ldy_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	lightDirectionZ_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	ldz_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	node : Light = None
	pass
class LightIntensityBPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : Light = None
	pass
class LightIntensityGPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : Light = None
	pass
class LightIntensityRPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : Light = None
	pass
class LightIntensityPlug(Plug):
	parent : LightDataPlug = PlugDescriptor("lightData")
	lightIntensityB_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lib_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lightIntensityG_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lig_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lightIntensityR_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	lir_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	node : Light = None
	pass
class LightShadowFractionPlug(Plug):
	parent : LightDataPlug = PlugDescriptor("lightData")
	node : Light = None
	pass
class LightSpecularPlug(Plug):
	parent : LightDataPlug = PlugDescriptor("lightData")
	node : Light = None
	pass
class PreShadowIntensityPlug(Plug):
	parent : LightDataPlug = PlugDescriptor("lightData")
	node : Light = None
	pass
class LightDataPlug(Plug):
	lightAmbient_ : LightAmbientPlug = PlugDescriptor("lightAmbient")
	la_ : LightAmbientPlug = PlugDescriptor("lightAmbient")
	lightBlindData_ : LightBlindDataPlug = PlugDescriptor("lightBlindData")
	lbl_ : LightBlindDataPlug = PlugDescriptor("lightBlindData")
	lightDiffuse_ : LightDiffusePlug = PlugDescriptor("lightDiffuse")
	ldf_ : LightDiffusePlug = PlugDescriptor("lightDiffuse")
	lightDirection_ : LightDirectionPlug = PlugDescriptor("lightDirection")
	ld_ : LightDirectionPlug = PlugDescriptor("lightDirection")
	lightIntensity_ : LightIntensityPlug = PlugDescriptor("lightIntensity")
	li_ : LightIntensityPlug = PlugDescriptor("lightIntensity")
	lightShadowFraction_ : LightShadowFractionPlug = PlugDescriptor("lightShadowFraction")
	lsf_ : LightShadowFractionPlug = PlugDescriptor("lightShadowFraction")
	lightSpecular_ : LightSpecularPlug = PlugDescriptor("lightSpecular")
	ls_ : LightSpecularPlug = PlugDescriptor("lightSpecular")
	preShadowIntensity_ : PreShadowIntensityPlug = PlugDescriptor("preShadowIntensity")
	psi_ : PreShadowIntensityPlug = PlugDescriptor("preShadowIntensity")
	node : Light = None
	pass
class LocatorScalePlug(Plug):
	node : Light = None
	pass
class MatrixEyeToWorldPlug(Plug):
	node : Light = None
	pass
class MatrixWorldToEyePlug(Plug):
	node : Light = None
	pass
class ObjectIdPlug(Plug):
	node : Light = None
	pass
class OpticalFXvisibilityBPlug(Plug):
	parent : OpticalFXvisibilityPlug = PlugDescriptor("opticalFXvisibility")
	node : Light = None
	pass
class OpticalFXvisibilityGPlug(Plug):
	parent : OpticalFXvisibilityPlug = PlugDescriptor("opticalFXvisibility")
	node : Light = None
	pass
class OpticalFXvisibilityRPlug(Plug):
	parent : OpticalFXvisibilityPlug = PlugDescriptor("opticalFXvisibility")
	node : Light = None
	pass
class OpticalFXvisibilityPlug(Plug):
	opticalFXvisibilityB_ : OpticalFXvisibilityBPlug = PlugDescriptor("opticalFXvisibilityB")
	ovb_ : OpticalFXvisibilityBPlug = PlugDescriptor("opticalFXvisibilityB")
	opticalFXvisibilityG_ : OpticalFXvisibilityGPlug = PlugDescriptor("opticalFXvisibilityG")
	ovg_ : OpticalFXvisibilityGPlug = PlugDescriptor("opticalFXvisibilityG")
	opticalFXvisibilityR_ : OpticalFXvisibilityRPlug = PlugDescriptor("opticalFXvisibilityR")
	ovr_ : OpticalFXvisibilityRPlug = PlugDescriptor("opticalFXvisibilityR")
	node : Light = None
	pass
class PointCameraXPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : Light = None
	pass
class PointCameraYPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : Light = None
	pass
class PointCameraZPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : Light = None
	pass
class PointCameraPlug(Plug):
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	px_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	py_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pz_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	node : Light = None
	pass
class PrimitiveIdPlug(Plug):
	node : Light = None
	pass
class RayDepthPlug(Plug):
	node : Light = None
	pass
class RayDepthLimitPlug(Plug):
	node : Light = None
	pass
class RaySamplerPlug(Plug):
	node : Light = None
	pass
class RenderStatePlug(Plug):
	node : Light = None
	pass
class ShadColorBPlug(Plug):
	parent : ShadowColorPlug = PlugDescriptor("shadowColor")
	node : Light = None
	pass
class ShadColorGPlug(Plug):
	parent : ShadowColorPlug = PlugDescriptor("shadowColor")
	node : Light = None
	pass
class ShadColorRPlug(Plug):
	parent : ShadowColorPlug = PlugDescriptor("shadowColor")
	node : Light = None
	pass
class ShadowColorPlug(Plug):
	shadColorB_ : ShadColorBPlug = PlugDescriptor("shadColorB")
	scb_ : ShadColorBPlug = PlugDescriptor("shadColorB")
	shadColorG_ : ShadColorGPlug = PlugDescriptor("shadColorG")
	scg_ : ShadColorGPlug = PlugDescriptor("shadColorG")
	shadColorR_ : ShadColorRPlug = PlugDescriptor("shadColorR")
	scr_ : ShadColorRPlug = PlugDescriptor("shadColorR")
	node : Light = None
	pass
class ShadowRaysPlug(Plug):
	node : Light = None
	pass
class UseRayTraceShadowsPlug(Plug):
	node : Light = None
	pass
class UCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : Light = None
	pass
class VCoordPlug(Plug):
	parent : UvCoordPlug = PlugDescriptor("uvCoord")
	node : Light = None
	pass
class UvCoordPlug(Plug):
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	uu_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	vv_ : VCoordPlug = PlugDescriptor("vCoord")
	node : Light = None
	pass
class UvFilterSizeXPlug(Plug):
	parent : UvFilterSizePlug = PlugDescriptor("uvFilterSize")
	node : Light = None
	pass
class UvFilterSizeYPlug(Plug):
	parent : UvFilterSizePlug = PlugDescriptor("uvFilterSize")
	node : Light = None
	pass
class UvFilterSizePlug(Plug):
	uvFilterSizeX_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	fsx_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	uvFilterSizeY_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	fsy_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	node : Light = None
	pass
# endregion


# define node class
class Light(Shape):
	centerOfIllumination_ : CenterOfIlluminationPlug = PlugDescriptor("centerOfIllumination")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	color_ : ColorPlug = PlugDescriptor("color")
	infoBits_ : InfoBitsPlug = PlugDescriptor("infoBits")
	intensity_ : IntensityPlug = PlugDescriptor("intensity")
	lightAmbient_ : LightAmbientPlug = PlugDescriptor("lightAmbient")
	lightBlindData_ : LightBlindDataPlug = PlugDescriptor("lightBlindData")
	lightDiffuse_ : LightDiffusePlug = PlugDescriptor("lightDiffuse")
	lightDirectionX_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	lightDirectionY_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	lightDirectionZ_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	lightDirection_ : LightDirectionPlug = PlugDescriptor("lightDirection")
	lightIntensityB_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lightIntensityG_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lightIntensityR_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	lightIntensity_ : LightIntensityPlug = PlugDescriptor("lightIntensity")
	lightShadowFraction_ : LightShadowFractionPlug = PlugDescriptor("lightShadowFraction")
	lightSpecular_ : LightSpecularPlug = PlugDescriptor("lightSpecular")
	preShadowIntensity_ : PreShadowIntensityPlug = PlugDescriptor("preShadowIntensity")
	lightData_ : LightDataPlug = PlugDescriptor("lightData")
	locatorScale_ : LocatorScalePlug = PlugDescriptor("locatorScale")
	matrixEyeToWorld_ : MatrixEyeToWorldPlug = PlugDescriptor("matrixEyeToWorld")
	matrixWorldToEye_ : MatrixWorldToEyePlug = PlugDescriptor("matrixWorldToEye")
	objectId_ : ObjectIdPlug = PlugDescriptor("objectId")
	opticalFXvisibilityB_ : OpticalFXvisibilityBPlug = PlugDescriptor("opticalFXvisibilityB")
	opticalFXvisibilityG_ : OpticalFXvisibilityGPlug = PlugDescriptor("opticalFXvisibilityG")
	opticalFXvisibilityR_ : OpticalFXvisibilityRPlug = PlugDescriptor("opticalFXvisibilityR")
	opticalFXvisibility_ : OpticalFXvisibilityPlug = PlugDescriptor("opticalFXvisibility")
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pointCamera_ : PointCameraPlug = PlugDescriptor("pointCamera")
	primitiveId_ : PrimitiveIdPlug = PlugDescriptor("primitiveId")
	rayDepth_ : RayDepthPlug = PlugDescriptor("rayDepth")
	rayDepthLimit_ : RayDepthLimitPlug = PlugDescriptor("rayDepthLimit")
	raySampler_ : RaySamplerPlug = PlugDescriptor("raySampler")
	renderState_ : RenderStatePlug = PlugDescriptor("renderState")
	shadColorB_ : ShadColorBPlug = PlugDescriptor("shadColorB")
	shadColorG_ : ShadColorGPlug = PlugDescriptor("shadColorG")
	shadColorR_ : ShadColorRPlug = PlugDescriptor("shadColorR")
	shadowColor_ : ShadowColorPlug = PlugDescriptor("shadowColor")
	shadowRays_ : ShadowRaysPlug = PlugDescriptor("shadowRays")
	useRayTraceShadows_ : UseRayTraceShadowsPlug = PlugDescriptor("useRayTraceShadows")
	uCoord_ : UCoordPlug = PlugDescriptor("uCoord")
	vCoord_ : VCoordPlug = PlugDescriptor("vCoord")
	uvCoord_ : UvCoordPlug = PlugDescriptor("uvCoord")
	uvFilterSizeX_ : UvFilterSizeXPlug = PlugDescriptor("uvFilterSizeX")
	uvFilterSizeY_ : UvFilterSizeYPlug = PlugDescriptor("uvFilterSizeY")
	uvFilterSize_ : UvFilterSizePlug = PlugDescriptor("uvFilterSize")

	# node attributes

	typeName = "light"
	typeIdInt = 1280527941
	nodeLeafClassAttrs = ["centerOfIllumination", "colorB", "colorG", "colorR", "color", "infoBits", "intensity", "lightAmbient", "lightBlindData", "lightDiffuse", "lightDirectionX", "lightDirectionY", "lightDirectionZ", "lightDirection", "lightIntensityB", "lightIntensityG", "lightIntensityR", "lightIntensity", "lightShadowFraction", "lightSpecular", "preShadowIntensity", "lightData", "locatorScale", "matrixEyeToWorld", "matrixWorldToEye", "objectId", "opticalFXvisibilityB", "opticalFXvisibilityG", "opticalFXvisibilityR", "opticalFXvisibility", "pointCameraX", "pointCameraY", "pointCameraZ", "pointCamera", "primitiveId", "rayDepth", "rayDepthLimit", "raySampler", "renderState", "shadColorB", "shadColorG", "shadColorR", "shadowColor", "shadowRays", "useRayTraceShadows", "uCoord", "vCoord", "uvCoord", "uvFilterSizeX", "uvFilterSizeY", "uvFilterSize"]
	nodeLeafPlugs = ["centerOfIllumination", "color", "infoBits", "intensity", "lightData", "locatorScale", "matrixEyeToWorld", "matrixWorldToEye", "objectId", "opticalFXvisibility", "pointCamera", "primitiveId", "rayDepth", "rayDepthLimit", "raySampler", "renderState", "shadowColor", "shadowRays", "useRayTraceShadows", "uvCoord", "uvFilterSize"]
	pass

