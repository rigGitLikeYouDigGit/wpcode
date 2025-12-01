

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
class AmbientColorBPlug(Plug):
	parent : AmbientColorPlug = PlugDescriptor("ambientColor")
	node : OceanShader = None
	pass
class AmbientColorGPlug(Plug):
	parent : AmbientColorPlug = PlugDescriptor("ambientColor")
	node : OceanShader = None
	pass
class AmbientColorRPlug(Plug):
	parent : AmbientColorPlug = PlugDescriptor("ambientColor")
	node : OceanShader = None
	pass
class AmbientColorPlug(Plug):
	ambientColorB_ : AmbientColorBPlug = PlugDescriptor("ambientColorB")
	acb_ : AmbientColorBPlug = PlugDescriptor("ambientColorB")
	ambientColorG_ : AmbientColorGPlug = PlugDescriptor("ambientColorG")
	acg_ : AmbientColorGPlug = PlugDescriptor("ambientColorG")
	ambientColorR_ : AmbientColorRPlug = PlugDescriptor("ambientColorR")
	acr_ : AmbientColorRPlug = PlugDescriptor("ambientColorR")
	node : OceanShader = None
	pass
class BumpBlurPlug(Plug):
	node : OceanShader = None
	pass
class DiffusePlug(Plug):
	node : OceanShader = None
	pass
class DisplacementPlug(Plug):
	node : OceanShader = None
	pass
class EccentricityPlug(Plug):
	node : OceanShader = None
	pass
class Environment_ColorBPlug(Plug):
	parent : Environment_ColorPlug = PlugDescriptor("environment_Color")
	node : OceanShader = None
	pass
class Environment_ColorGPlug(Plug):
	parent : Environment_ColorPlug = PlugDescriptor("environment_Color")
	node : OceanShader = None
	pass
class Environment_ColorRPlug(Plug):
	parent : Environment_ColorPlug = PlugDescriptor("environment_Color")
	node : OceanShader = None
	pass
class Environment_ColorPlug(Plug):
	parent : EnvironmentPlug = PlugDescriptor("environment")
	environment_ColorB_ : Environment_ColorBPlug = PlugDescriptor("environment_ColorB")
	envcb_ : Environment_ColorBPlug = PlugDescriptor("environment_ColorB")
	environment_ColorG_ : Environment_ColorGPlug = PlugDescriptor("environment_ColorG")
	envcg_ : Environment_ColorGPlug = PlugDescriptor("environment_ColorG")
	environment_ColorR_ : Environment_ColorRPlug = PlugDescriptor("environment_ColorR")
	envcr_ : Environment_ColorRPlug = PlugDescriptor("environment_ColorR")
	node : OceanShader = None
	pass
class Environment_InterpPlug(Plug):
	parent : EnvironmentPlug = PlugDescriptor("environment")
	node : OceanShader = None
	pass
class Environment_PositionPlug(Plug):
	parent : EnvironmentPlug = PlugDescriptor("environment")
	node : OceanShader = None
	pass
class EnvironmentPlug(Plug):
	environment_Color_ : Environment_ColorPlug = PlugDescriptor("environment_Color")
	envc_ : Environment_ColorPlug = PlugDescriptor("environment_Color")
	environment_Interp_ : Environment_InterpPlug = PlugDescriptor("environment_Interp")
	envi_ : Environment_InterpPlug = PlugDescriptor("environment_Interp")
	environment_Position_ : Environment_PositionPlug = PlugDescriptor("environment_Position")
	envp_ : Environment_PositionPlug = PlugDescriptor("environment_Position")
	node : OceanShader = None
	pass
class FilterSizeXPlug(Plug):
	parent : FilterSizePlug = PlugDescriptor("filterSize")
	node : OceanShader = None
	pass
class FilterSizeYPlug(Plug):
	parent : FilterSizePlug = PlugDescriptor("filterSize")
	node : OceanShader = None
	pass
class FilterSizeZPlug(Plug):
	parent : FilterSizePlug = PlugDescriptor("filterSize")
	node : OceanShader = None
	pass
class FilterSizePlug(Plug):
	filterSizeX_ : FilterSizeXPlug = PlugDescriptor("filterSizeX")
	fsx_ : FilterSizeXPlug = PlugDescriptor("filterSizeX")
	filterSizeY_ : FilterSizeYPlug = PlugDescriptor("filterSizeY")
	fsy_ : FilterSizeYPlug = PlugDescriptor("filterSizeY")
	filterSizeZ_ : FilterSizeZPlug = PlugDescriptor("filterSizeZ")
	fsz_ : FilterSizeZPlug = PlugDescriptor("filterSizeZ")
	node : OceanShader = None
	pass
class FoamColorBPlug(Plug):
	parent : FoamColorPlug = PlugDescriptor("foamColor")
	node : OceanShader = None
	pass
class FoamColorGPlug(Plug):
	parent : FoamColorPlug = PlugDescriptor("foamColor")
	node : OceanShader = None
	pass
class FoamColorRPlug(Plug):
	parent : FoamColorPlug = PlugDescriptor("foamColor")
	node : OceanShader = None
	pass
class FoamColorPlug(Plug):
	foamColorB_ : FoamColorBPlug = PlugDescriptor("foamColorB")
	fcb_ : FoamColorBPlug = PlugDescriptor("foamColorB")
	foamColorG_ : FoamColorGPlug = PlugDescriptor("foamColorG")
	fcg_ : FoamColorGPlug = PlugDescriptor("foamColorG")
	foamColorR_ : FoamColorRPlug = PlugDescriptor("foamColorR")
	fcr_ : FoamColorRPlug = PlugDescriptor("foamColorR")
	node : OceanShader = None
	pass
class FoamEmissionPlug(Plug):
	node : OceanShader = None
	pass
class FoamOffsetPlug(Plug):
	node : OceanShader = None
	pass
class FoamThresholdPlug(Plug):
	node : OceanShader = None
	pass
class GlowIntensityPlug(Plug):
	node : OceanShader = None
	pass
class HorizonFilterPlug(Plug):
	node : OceanShader = None
	pass
class IncandescenceBPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : OceanShader = None
	pass
class IncandescenceGPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : OceanShader = None
	pass
class IncandescenceRPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : OceanShader = None
	pass
class IncandescencePlug(Plug):
	incandescenceB_ : IncandescenceBPlug = PlugDescriptor("incandescenceB")
	ib_ : IncandescenceBPlug = PlugDescriptor("incandescenceB")
	incandescenceG_ : IncandescenceGPlug = PlugDescriptor("incandescenceG")
	ig_ : IncandescenceGPlug = PlugDescriptor("incandescenceG")
	incandescenceR_ : IncandescenceRPlug = PlugDescriptor("incandescenceR")
	ir_ : IncandescenceRPlug = PlugDescriptor("incandescenceR")
	node : OceanShader = None
	pass
class LightAmbientPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : OceanShader = None
	pass
class LightBlindDataPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : OceanShader = None
	pass
class LightDiffusePlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : OceanShader = None
	pass
class LightDirectionXPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : OceanShader = None
	pass
class LightDirectionYPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : OceanShader = None
	pass
class LightDirectionZPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : OceanShader = None
	pass
class LightDirectionPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	lightDirectionX_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	ldx_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	lightDirectionY_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	ldy_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	lightDirectionZ_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	ldz_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	node : OceanShader = None
	pass
class LightIntensityBPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : OceanShader = None
	pass
class LightIntensityGPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : OceanShader = None
	pass
class LightIntensityRPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : OceanShader = None
	pass
class LightIntensityPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	lightIntensityB_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lib_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lightIntensityG_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lig_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lightIntensityR_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	lir_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	node : OceanShader = None
	pass
class LightShadowFractionPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : OceanShader = None
	pass
class LightSpecularPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : OceanShader = None
	pass
class PreShadowIntensityPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : OceanShader = None
	pass
class LightDataArrayPlug(Plug):
	lightAmbient_ : LightAmbientPlug = PlugDescriptor("lightAmbient")
	la_ : LightAmbientPlug = PlugDescriptor("lightAmbient")
	lightBlindData_ : LightBlindDataPlug = PlugDescriptor("lightBlindData")
	lbd_ : LightBlindDataPlug = PlugDescriptor("lightBlindData")
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
	node : OceanShader = None
	pass
class MatrixEyeToWorldPlug(Plug):
	node : OceanShader = None
	pass
class MatrixWorldToEyePlug(Plug):
	node : OceanShader = None
	pass
class MatteOpacityPlug(Plug):
	node : OceanShader = None
	pass
class MatteOpacityModePlug(Plug):
	node : OceanShader = None
	pass
class MediumRefractiveIndexPlug(Plug):
	node : OceanShader = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : OceanShader = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : OceanShader = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : OceanShader = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	nx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ny_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	nz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : OceanShader = None
	pass
class NumFrequenciesPlug(Plug):
	node : OceanShader = None
	pass
class ObjectIdPlug(Plug):
	node : OceanShader = None
	pass
class ObserverSpeedPlug(Plug):
	node : OceanShader = None
	pass
class OpacityDepthPlug(Plug):
	node : OceanShader = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : OceanShader = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : OceanShader = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : OceanShader = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : OceanShader = None
	pass
class OutFoamPlug(Plug):
	node : OceanShader = None
	pass
class OutGlowColorBPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : OceanShader = None
	pass
class OutGlowColorGPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : OceanShader = None
	pass
class OutGlowColorRPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : OceanShader = None
	pass
class OutGlowColorPlug(Plug):
	outGlowColorB_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	ogb_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	outGlowColorG_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	ogg_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	outGlowColorR_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	ogr_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	node : OceanShader = None
	pass
class OutMatteOpacityBPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : OceanShader = None
	pass
class OutMatteOpacityGPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : OceanShader = None
	pass
class OutMatteOpacityRPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : OceanShader = None
	pass
class OutMatteOpacityPlug(Plug):
	outMatteOpacityB_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	omob_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	outMatteOpacityG_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	omog_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	outMatteOpacityR_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	omor_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	node : OceanShader = None
	pass
class OutTransparencyBPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : OceanShader = None
	pass
class OutTransparencyGPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : OceanShader = None
	pass
class OutTransparencyRPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : OceanShader = None
	pass
class OutTransparencyPlug(Plug):
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	otb_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	otg_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	otr_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	node : OceanShader = None
	pass
class PointCameraXPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : OceanShader = None
	pass
class PointCameraYPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : OceanShader = None
	pass
class PointCameraZPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : OceanShader = None
	pass
class PointCameraPlug(Plug):
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	px_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	py_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pz_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	node : OceanShader = None
	pass
class PrimitiveIdPlug(Plug):
	node : OceanShader = None
	pass
class RayDepthPlug(Plug):
	node : OceanShader = None
	pass
class RayDirectionXPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : OceanShader = None
	pass
class RayDirectionYPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : OceanShader = None
	pass
class RayDirectionZPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : OceanShader = None
	pass
class RayDirectionPlug(Plug):
	rayDirectionX_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rdx_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rayDirectionY_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rdy_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rayDirectionZ_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	rdz_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	node : OceanShader = None
	pass
class RayInstancePlug(Plug):
	node : OceanShader = None
	pass
class RaySamplerPlug(Plug):
	node : OceanShader = None
	pass
class RefPointCameraXPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : OceanShader = None
	pass
class RefPointCameraYPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : OceanShader = None
	pass
class RefPointCameraZPlug(Plug):
	parent : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	node : OceanShader = None
	pass
class RefPointCameraPlug(Plug):
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	rcx_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	rcy_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	rcz_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	node : OceanShader = None
	pass
class ReflectedColorBPlug(Plug):
	parent : ReflectedColorPlug = PlugDescriptor("reflectedColor")
	node : OceanShader = None
	pass
class ReflectedColorGPlug(Plug):
	parent : ReflectedColorPlug = PlugDescriptor("reflectedColor")
	node : OceanShader = None
	pass
class ReflectedColorRPlug(Plug):
	parent : ReflectedColorPlug = PlugDescriptor("reflectedColor")
	node : OceanShader = None
	pass
class ReflectedColorPlug(Plug):
	reflectedColorB_ : ReflectedColorBPlug = PlugDescriptor("reflectedColorB")
	rb_ : ReflectedColorBPlug = PlugDescriptor("reflectedColorB")
	reflectedColorG_ : ReflectedColorGPlug = PlugDescriptor("reflectedColorG")
	rg_ : ReflectedColorGPlug = PlugDescriptor("reflectedColorG")
	reflectedColorR_ : ReflectedColorRPlug = PlugDescriptor("reflectedColorR")
	rr_ : ReflectedColorRPlug = PlugDescriptor("reflectedColorR")
	node : OceanShader = None
	pass
class ReflectionLimitPlug(Plug):
	node : OceanShader = None
	pass
class ReflectionSpecularityPlug(Plug):
	node : OceanShader = None
	pass
class ReflectivityPlug(Plug):
	node : OceanShader = None
	pass
class RefractionLimitPlug(Plug):
	node : OceanShader = None
	pass
class RefractionsPlug(Plug):
	node : OceanShader = None
	pass
class RefractiveIndexPlug(Plug):
	node : OceanShader = None
	pass
class ScalePlug(Plug):
	node : OceanShader = None
	pass
class ShadowAttenuationPlug(Plug):
	node : OceanShader = None
	pass
class SpecularColorBPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : OceanShader = None
	pass
class SpecularColorGPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : OceanShader = None
	pass
class SpecularColorRPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : OceanShader = None
	pass
class SpecularColorPlug(Plug):
	specularColorB_ : SpecularColorBPlug = PlugDescriptor("specularColorB")
	sb_ : SpecularColorBPlug = PlugDescriptor("specularColorB")
	specularColorG_ : SpecularColorGPlug = PlugDescriptor("specularColorG")
	sg_ : SpecularColorGPlug = PlugDescriptor("specularColorG")
	specularColorR_ : SpecularColorRPlug = PlugDescriptor("specularColorR")
	sr_ : SpecularColorRPlug = PlugDescriptor("specularColorR")
	node : OceanShader = None
	pass
class SpecularGlowPlug(Plug):
	node : OceanShader = None
	pass
class SpecularityPlug(Plug):
	node : OceanShader = None
	pass
class TimePlug(Plug):
	node : OceanShader = None
	pass
class TranslucencePlug(Plug):
	node : OceanShader = None
	pass
class TranslucenceDepthPlug(Plug):
	node : OceanShader = None
	pass
class TranslucenceFocusPlug(Plug):
	node : OceanShader = None
	pass
class TransparencyBPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : OceanShader = None
	pass
class TransparencyGPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : OceanShader = None
	pass
class TransparencyRPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : OceanShader = None
	pass
class TransparencyPlug(Plug):
	transparencyB_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	itb_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	transparencyG_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	itg_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	transparencyR_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	itr_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	node : OceanShader = None
	pass
class TriangleNormalCameraXPlug(Plug):
	parent : TriangleNormalCameraPlug = PlugDescriptor("triangleNormalCamera")
	node : OceanShader = None
	pass
class TriangleNormalCameraYPlug(Plug):
	parent : TriangleNormalCameraPlug = PlugDescriptor("triangleNormalCamera")
	node : OceanShader = None
	pass
class TriangleNormalCameraZPlug(Plug):
	parent : TriangleNormalCameraPlug = PlugDescriptor("triangleNormalCamera")
	node : OceanShader = None
	pass
class TriangleNormalCameraPlug(Plug):
	triangleNormalCameraX_ : TriangleNormalCameraXPlug = PlugDescriptor("triangleNormalCameraX")
	tnx_ : TriangleNormalCameraXPlug = PlugDescriptor("triangleNormalCameraX")
	triangleNormalCameraY_ : TriangleNormalCameraYPlug = PlugDescriptor("triangleNormalCameraY")
	tny_ : TriangleNormalCameraYPlug = PlugDescriptor("triangleNormalCameraY")
	triangleNormalCameraZ_ : TriangleNormalCameraZPlug = PlugDescriptor("triangleNormalCameraZ")
	tnz_ : TriangleNormalCameraZPlug = PlugDescriptor("triangleNormalCameraZ")
	node : OceanShader = None
	pass
class TroughShadowingPlug(Plug):
	node : OceanShader = None
	pass
class WaterColorBPlug(Plug):
	parent : WaterColorPlug = PlugDescriptor("waterColor")
	node : OceanShader = None
	pass
class WaterColorGPlug(Plug):
	parent : WaterColorPlug = PlugDescriptor("waterColor")
	node : OceanShader = None
	pass
class WaterColorRPlug(Plug):
	parent : WaterColorPlug = PlugDescriptor("waterColor")
	node : OceanShader = None
	pass
class WaterColorPlug(Plug):
	waterColorB_ : WaterColorBPlug = PlugDescriptor("waterColorB")
	wcb_ : WaterColorBPlug = PlugDescriptor("waterColorB")
	waterColorG_ : WaterColorGPlug = PlugDescriptor("waterColorG")
	wcg_ : WaterColorGPlug = PlugDescriptor("waterColorG")
	waterColorR_ : WaterColorRPlug = PlugDescriptor("waterColorR")
	wcr_ : WaterColorRPlug = PlugDescriptor("waterColorR")
	node : OceanShader = None
	pass
class WaveDirSpreadPlug(Plug):
	node : OceanShader = None
	pass
class WaveHeight_FloatValuePlug(Plug):
	parent : WaveHeightPlug = PlugDescriptor("waveHeight")
	node : OceanShader = None
	pass
class WaveHeight_InterpPlug(Plug):
	parent : WaveHeightPlug = PlugDescriptor("waveHeight")
	node : OceanShader = None
	pass
class WaveHeight_PositionPlug(Plug):
	parent : WaveHeightPlug = PlugDescriptor("waveHeight")
	node : OceanShader = None
	pass
class WaveHeightPlug(Plug):
	waveHeight_FloatValue_ : WaveHeight_FloatValuePlug = PlugDescriptor("waveHeight_FloatValue")
	whfv_ : WaveHeight_FloatValuePlug = PlugDescriptor("waveHeight_FloatValue")
	waveHeight_Interp_ : WaveHeight_InterpPlug = PlugDescriptor("waveHeight_Interp")
	whi_ : WaveHeight_InterpPlug = PlugDescriptor("waveHeight_Interp")
	waveHeight_Position_ : WaveHeight_PositionPlug = PlugDescriptor("waveHeight_Position")
	whp_ : WaveHeight_PositionPlug = PlugDescriptor("waveHeight_Position")
	node : OceanShader = None
	pass
class WaveHeightOffsetPlug(Plug):
	node : OceanShader = None
	pass
class WaveLengthMaxPlug(Plug):
	node : OceanShader = None
	pass
class WaveLengthMinPlug(Plug):
	node : OceanShader = None
	pass
class WavePeaking_FloatValuePlug(Plug):
	parent : WavePeakingPlug = PlugDescriptor("wavePeaking")
	node : OceanShader = None
	pass
class WavePeaking_InterpPlug(Plug):
	parent : WavePeakingPlug = PlugDescriptor("wavePeaking")
	node : OceanShader = None
	pass
class WavePeaking_PositionPlug(Plug):
	parent : WavePeakingPlug = PlugDescriptor("wavePeaking")
	node : OceanShader = None
	pass
class WavePeakingPlug(Plug):
	wavePeaking_FloatValue_ : WavePeaking_FloatValuePlug = PlugDescriptor("wavePeaking_FloatValue")
	wpfv_ : WavePeaking_FloatValuePlug = PlugDescriptor("wavePeaking_FloatValue")
	wavePeaking_Interp_ : WavePeaking_InterpPlug = PlugDescriptor("wavePeaking_Interp")
	wpi_ : WavePeaking_InterpPlug = PlugDescriptor("wavePeaking_Interp")
	wavePeaking_Position_ : WavePeaking_PositionPlug = PlugDescriptor("wavePeaking_Position")
	wpp_ : WavePeaking_PositionPlug = PlugDescriptor("wavePeaking_Position")
	node : OceanShader = None
	pass
class WaveSpeedPlug(Plug):
	node : OceanShader = None
	pass
class WaveTurbulence_FloatValuePlug(Plug):
	parent : WaveTurbulencePlug = PlugDescriptor("waveTurbulence")
	node : OceanShader = None
	pass
class WaveTurbulence_InterpPlug(Plug):
	parent : WaveTurbulencePlug = PlugDescriptor("waveTurbulence")
	node : OceanShader = None
	pass
class WaveTurbulence_PositionPlug(Plug):
	parent : WaveTurbulencePlug = PlugDescriptor("waveTurbulence")
	node : OceanShader = None
	pass
class WaveTurbulencePlug(Plug):
	waveTurbulence_FloatValue_ : WaveTurbulence_FloatValuePlug = PlugDescriptor("waveTurbulence_FloatValue")
	wtbfv_ : WaveTurbulence_FloatValuePlug = PlugDescriptor("waveTurbulence_FloatValue")
	waveTurbulence_Interp_ : WaveTurbulence_InterpPlug = PlugDescriptor("waveTurbulence_Interp")
	wtbi_ : WaveTurbulence_InterpPlug = PlugDescriptor("waveTurbulence_Interp")
	waveTurbulence_Position_ : WaveTurbulence_PositionPlug = PlugDescriptor("waveTurbulence_Position")
	wtbp_ : WaveTurbulence_PositionPlug = PlugDescriptor("waveTurbulence_Position")
	node : OceanShader = None
	pass
class WindUPlug(Plug):
	parent : WindUVPlug = PlugDescriptor("windUV")
	node : OceanShader = None
	pass
class WindVPlug(Plug):
	parent : WindUVPlug = PlugDescriptor("windUV")
	node : OceanShader = None
	pass
class WindUVPlug(Plug):
	windU_ : WindUPlug = PlugDescriptor("windU")
	wiu_ : WindUPlug = PlugDescriptor("windU")
	windV_ : WindVPlug = PlugDescriptor("windV")
	wiv_ : WindVPlug = PlugDescriptor("windV")
	node : OceanShader = None
	pass
# endregion


# define node class
class OceanShader(ShadingDependNode):
	ambientColorB_ : AmbientColorBPlug = PlugDescriptor("ambientColorB")
	ambientColorG_ : AmbientColorGPlug = PlugDescriptor("ambientColorG")
	ambientColorR_ : AmbientColorRPlug = PlugDescriptor("ambientColorR")
	ambientColor_ : AmbientColorPlug = PlugDescriptor("ambientColor")
	bumpBlur_ : BumpBlurPlug = PlugDescriptor("bumpBlur")
	diffuse_ : DiffusePlug = PlugDescriptor("diffuse")
	displacement_ : DisplacementPlug = PlugDescriptor("displacement")
	eccentricity_ : EccentricityPlug = PlugDescriptor("eccentricity")
	environment_ColorB_ : Environment_ColorBPlug = PlugDescriptor("environment_ColorB")
	environment_ColorG_ : Environment_ColorGPlug = PlugDescriptor("environment_ColorG")
	environment_ColorR_ : Environment_ColorRPlug = PlugDescriptor("environment_ColorR")
	environment_Color_ : Environment_ColorPlug = PlugDescriptor("environment_Color")
	environment_Interp_ : Environment_InterpPlug = PlugDescriptor("environment_Interp")
	environment_Position_ : Environment_PositionPlug = PlugDescriptor("environment_Position")
	environment_ : EnvironmentPlug = PlugDescriptor("environment")
	filterSizeX_ : FilterSizeXPlug = PlugDescriptor("filterSizeX")
	filterSizeY_ : FilterSizeYPlug = PlugDescriptor("filterSizeY")
	filterSizeZ_ : FilterSizeZPlug = PlugDescriptor("filterSizeZ")
	filterSize_ : FilterSizePlug = PlugDescriptor("filterSize")
	foamColorB_ : FoamColorBPlug = PlugDescriptor("foamColorB")
	foamColorG_ : FoamColorGPlug = PlugDescriptor("foamColorG")
	foamColorR_ : FoamColorRPlug = PlugDescriptor("foamColorR")
	foamColor_ : FoamColorPlug = PlugDescriptor("foamColor")
	foamEmission_ : FoamEmissionPlug = PlugDescriptor("foamEmission")
	foamOffset_ : FoamOffsetPlug = PlugDescriptor("foamOffset")
	foamThreshold_ : FoamThresholdPlug = PlugDescriptor("foamThreshold")
	glowIntensity_ : GlowIntensityPlug = PlugDescriptor("glowIntensity")
	horizonFilter_ : HorizonFilterPlug = PlugDescriptor("horizonFilter")
	incandescenceB_ : IncandescenceBPlug = PlugDescriptor("incandescenceB")
	incandescenceG_ : IncandescenceGPlug = PlugDescriptor("incandescenceG")
	incandescenceR_ : IncandescenceRPlug = PlugDescriptor("incandescenceR")
	incandescence_ : IncandescencePlug = PlugDescriptor("incandescence")
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
	lightDataArray_ : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	matrixEyeToWorld_ : MatrixEyeToWorldPlug = PlugDescriptor("matrixEyeToWorld")
	matrixWorldToEye_ : MatrixWorldToEyePlug = PlugDescriptor("matrixWorldToEye")
	matteOpacity_ : MatteOpacityPlug = PlugDescriptor("matteOpacity")
	matteOpacityMode_ : MatteOpacityModePlug = PlugDescriptor("matteOpacityMode")
	mediumRefractiveIndex_ : MediumRefractiveIndexPlug = PlugDescriptor("mediumRefractiveIndex")
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	normalCamera_ : NormalCameraPlug = PlugDescriptor("normalCamera")
	numFrequencies_ : NumFrequenciesPlug = PlugDescriptor("numFrequencies")
	objectId_ : ObjectIdPlug = PlugDescriptor("objectId")
	observerSpeed_ : ObserverSpeedPlug = PlugDescriptor("observerSpeed")
	opacityDepth_ : OpacityDepthPlug = PlugDescriptor("opacityDepth")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	outFoam_ : OutFoamPlug = PlugDescriptor("outFoam")
	outGlowColorB_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	outGlowColorG_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	outGlowColorR_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	outGlowColor_ : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	outMatteOpacityB_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	outMatteOpacityG_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	outMatteOpacityR_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	outMatteOpacity_ : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	outTransparency_ : OutTransparencyPlug = PlugDescriptor("outTransparency")
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pointCamera_ : PointCameraPlug = PlugDescriptor("pointCamera")
	primitiveId_ : PrimitiveIdPlug = PlugDescriptor("primitiveId")
	rayDepth_ : RayDepthPlug = PlugDescriptor("rayDepth")
	rayDirectionX_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rayDirectionY_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rayDirectionZ_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	rayDirection_ : RayDirectionPlug = PlugDescriptor("rayDirection")
	rayInstance_ : RayInstancePlug = PlugDescriptor("rayInstance")
	raySampler_ : RaySamplerPlug = PlugDescriptor("raySampler")
	refPointCameraX_ : RefPointCameraXPlug = PlugDescriptor("refPointCameraX")
	refPointCameraY_ : RefPointCameraYPlug = PlugDescriptor("refPointCameraY")
	refPointCameraZ_ : RefPointCameraZPlug = PlugDescriptor("refPointCameraZ")
	refPointCamera_ : RefPointCameraPlug = PlugDescriptor("refPointCamera")
	reflectedColorB_ : ReflectedColorBPlug = PlugDescriptor("reflectedColorB")
	reflectedColorG_ : ReflectedColorGPlug = PlugDescriptor("reflectedColorG")
	reflectedColorR_ : ReflectedColorRPlug = PlugDescriptor("reflectedColorR")
	reflectedColor_ : ReflectedColorPlug = PlugDescriptor("reflectedColor")
	reflectionLimit_ : ReflectionLimitPlug = PlugDescriptor("reflectionLimit")
	reflectionSpecularity_ : ReflectionSpecularityPlug = PlugDescriptor("reflectionSpecularity")
	reflectivity_ : ReflectivityPlug = PlugDescriptor("reflectivity")
	refractionLimit_ : RefractionLimitPlug = PlugDescriptor("refractionLimit")
	refractions_ : RefractionsPlug = PlugDescriptor("refractions")
	refractiveIndex_ : RefractiveIndexPlug = PlugDescriptor("refractiveIndex")
	scale_ : ScalePlug = PlugDescriptor("scale")
	shadowAttenuation_ : ShadowAttenuationPlug = PlugDescriptor("shadowAttenuation")
	specularColorB_ : SpecularColorBPlug = PlugDescriptor("specularColorB")
	specularColorG_ : SpecularColorGPlug = PlugDescriptor("specularColorG")
	specularColorR_ : SpecularColorRPlug = PlugDescriptor("specularColorR")
	specularColor_ : SpecularColorPlug = PlugDescriptor("specularColor")
	specularGlow_ : SpecularGlowPlug = PlugDescriptor("specularGlow")
	specularity_ : SpecularityPlug = PlugDescriptor("specularity")
	time_ : TimePlug = PlugDescriptor("time")
	translucence_ : TranslucencePlug = PlugDescriptor("translucence")
	translucenceDepth_ : TranslucenceDepthPlug = PlugDescriptor("translucenceDepth")
	translucenceFocus_ : TranslucenceFocusPlug = PlugDescriptor("translucenceFocus")
	transparencyB_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	transparencyG_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	transparencyR_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	transparency_ : TransparencyPlug = PlugDescriptor("transparency")
	triangleNormalCameraX_ : TriangleNormalCameraXPlug = PlugDescriptor("triangleNormalCameraX")
	triangleNormalCameraY_ : TriangleNormalCameraYPlug = PlugDescriptor("triangleNormalCameraY")
	triangleNormalCameraZ_ : TriangleNormalCameraZPlug = PlugDescriptor("triangleNormalCameraZ")
	triangleNormalCamera_ : TriangleNormalCameraPlug = PlugDescriptor("triangleNormalCamera")
	troughShadowing_ : TroughShadowingPlug = PlugDescriptor("troughShadowing")
	waterColorB_ : WaterColorBPlug = PlugDescriptor("waterColorB")
	waterColorG_ : WaterColorGPlug = PlugDescriptor("waterColorG")
	waterColorR_ : WaterColorRPlug = PlugDescriptor("waterColorR")
	waterColor_ : WaterColorPlug = PlugDescriptor("waterColor")
	waveDirSpread_ : WaveDirSpreadPlug = PlugDescriptor("waveDirSpread")
	waveHeight_FloatValue_ : WaveHeight_FloatValuePlug = PlugDescriptor("waveHeight_FloatValue")
	waveHeight_Interp_ : WaveHeight_InterpPlug = PlugDescriptor("waveHeight_Interp")
	waveHeight_Position_ : WaveHeight_PositionPlug = PlugDescriptor("waveHeight_Position")
	waveHeight_ : WaveHeightPlug = PlugDescriptor("waveHeight")
	waveHeightOffset_ : WaveHeightOffsetPlug = PlugDescriptor("waveHeightOffset")
	waveLengthMax_ : WaveLengthMaxPlug = PlugDescriptor("waveLengthMax")
	waveLengthMin_ : WaveLengthMinPlug = PlugDescriptor("waveLengthMin")
	wavePeaking_FloatValue_ : WavePeaking_FloatValuePlug = PlugDescriptor("wavePeaking_FloatValue")
	wavePeaking_Interp_ : WavePeaking_InterpPlug = PlugDescriptor("wavePeaking_Interp")
	wavePeaking_Position_ : WavePeaking_PositionPlug = PlugDescriptor("wavePeaking_Position")
	wavePeaking_ : WavePeakingPlug = PlugDescriptor("wavePeaking")
	waveSpeed_ : WaveSpeedPlug = PlugDescriptor("waveSpeed")
	waveTurbulence_FloatValue_ : WaveTurbulence_FloatValuePlug = PlugDescriptor("waveTurbulence_FloatValue")
	waveTurbulence_Interp_ : WaveTurbulence_InterpPlug = PlugDescriptor("waveTurbulence_Interp")
	waveTurbulence_Position_ : WaveTurbulence_PositionPlug = PlugDescriptor("waveTurbulence_Position")
	waveTurbulence_ : WaveTurbulencePlug = PlugDescriptor("waveTurbulence")
	windU_ : WindUPlug = PlugDescriptor("windU")
	windV_ : WindVPlug = PlugDescriptor("windV")
	windUV_ : WindUVPlug = PlugDescriptor("windUV")

	# node attributes

	typeName = "oceanShader"
	apiTypeInt = 898
	apiTypeStr = "kOceanShader"
	typeIdInt = 1380929619
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["ambientColorB", "ambientColorG", "ambientColorR", "ambientColor", "bumpBlur", "diffuse", "displacement", "eccentricity", "environment_ColorB", "environment_ColorG", "environment_ColorR", "environment_Color", "environment_Interp", "environment_Position", "environment", "filterSizeX", "filterSizeY", "filterSizeZ", "filterSize", "foamColorB", "foamColorG", "foamColorR", "foamColor", "foamEmission", "foamOffset", "foamThreshold", "glowIntensity", "horizonFilter", "incandescenceB", "incandescenceG", "incandescenceR", "incandescence", "lightAmbient", "lightBlindData", "lightDiffuse", "lightDirectionX", "lightDirectionY", "lightDirectionZ", "lightDirection", "lightIntensityB", "lightIntensityG", "lightIntensityR", "lightIntensity", "lightShadowFraction", "lightSpecular", "preShadowIntensity", "lightDataArray", "matrixEyeToWorld", "matrixWorldToEye", "matteOpacity", "matteOpacityMode", "mediumRefractiveIndex", "normalCameraX", "normalCameraY", "normalCameraZ", "normalCamera", "numFrequencies", "objectId", "observerSpeed", "opacityDepth", "outColorB", "outColorG", "outColorR", "outColor", "outFoam", "outGlowColorB", "outGlowColorG", "outGlowColorR", "outGlowColor", "outMatteOpacityB", "outMatteOpacityG", "outMatteOpacityR", "outMatteOpacity", "outTransparencyB", "outTransparencyG", "outTransparencyR", "outTransparency", "pointCameraX", "pointCameraY", "pointCameraZ", "pointCamera", "primitiveId", "rayDepth", "rayDirectionX", "rayDirectionY", "rayDirectionZ", "rayDirection", "rayInstance", "raySampler", "refPointCameraX", "refPointCameraY", "refPointCameraZ", "refPointCamera", "reflectedColorB", "reflectedColorG", "reflectedColorR", "reflectedColor", "reflectionLimit", "reflectionSpecularity", "reflectivity", "refractionLimit", "refractions", "refractiveIndex", "scale", "shadowAttenuation", "specularColorB", "specularColorG", "specularColorR", "specularColor", "specularGlow", "specularity", "time", "translucence", "translucenceDepth", "translucenceFocus", "transparencyB", "transparencyG", "transparencyR", "transparency", "triangleNormalCameraX", "triangleNormalCameraY", "triangleNormalCameraZ", "triangleNormalCamera", "troughShadowing", "waterColorB", "waterColorG", "waterColorR", "waterColor", "waveDirSpread", "waveHeight_FloatValue", "waveHeight_Interp", "waveHeight_Position", "waveHeight", "waveHeightOffset", "waveLengthMax", "waveLengthMin", "wavePeaking_FloatValue", "wavePeaking_Interp", "wavePeaking_Position", "wavePeaking", "waveSpeed", "waveTurbulence_FloatValue", "waveTurbulence_Interp", "waveTurbulence_Position", "waveTurbulence", "windU", "windV", "windUV"]
	nodeLeafPlugs = ["ambientColor", "bumpBlur", "diffuse", "displacement", "eccentricity", "environment", "filterSize", "foamColor", "foamEmission", "foamOffset", "foamThreshold", "glowIntensity", "horizonFilter", "incandescence", "lightDataArray", "matrixEyeToWorld", "matrixWorldToEye", "matteOpacity", "matteOpacityMode", "mediumRefractiveIndex", "normalCamera", "numFrequencies", "objectId", "observerSpeed", "opacityDepth", "outColor", "outFoam", "outGlowColor", "outMatteOpacity", "outTransparency", "pointCamera", "primitiveId", "rayDepth", "rayDirection", "rayInstance", "raySampler", "refPointCamera", "reflectedColor", "reflectionLimit", "reflectionSpecularity", "reflectivity", "refractionLimit", "refractions", "refractiveIndex", "scale", "shadowAttenuation", "specularColor", "specularGlow", "specularity", "time", "translucence", "translucenceDepth", "translucenceFocus", "transparency", "triangleNormalCamera", "troughShadowing", "waterColor", "waveDirSpread", "waveHeight", "waveHeightOffset", "waveLengthMax", "waveLengthMin", "wavePeaking", "waveSpeed", "waveTurbulence", "windUV"]
	pass

