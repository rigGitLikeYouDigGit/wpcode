

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
	node : RampShader = None
	pass
class AmbientColorGPlug(Plug):
	parent : AmbientColorPlug = PlugDescriptor("ambientColor")
	node : RampShader = None
	pass
class AmbientColorRPlug(Plug):
	parent : AmbientColorPlug = PlugDescriptor("ambientColor")
	node : RampShader = None
	pass
class AmbientColorPlug(Plug):
	ambientColorB_ : AmbientColorBPlug = PlugDescriptor("ambientColorB")
	acb_ : AmbientColorBPlug = PlugDescriptor("ambientColorB")
	ambientColorG_ : AmbientColorGPlug = PlugDescriptor("ambientColorG")
	acg_ : AmbientColorGPlug = PlugDescriptor("ambientColorG")
	ambientColorR_ : AmbientColorRPlug = PlugDescriptor("ambientColorR")
	acr_ : AmbientColorRPlug = PlugDescriptor("ambientColorR")
	node : RampShader = None
	pass
class ChromaticAberrationPlug(Plug):
	node : RampShader = None
	pass
class Color_ColorBPlug(Plug):
	parent : Color_ColorPlug = PlugDescriptor("color_Color")
	node : RampShader = None
	pass
class Color_ColorGPlug(Plug):
	parent : Color_ColorPlug = PlugDescriptor("color_Color")
	node : RampShader = None
	pass
class Color_ColorRPlug(Plug):
	parent : Color_ColorPlug = PlugDescriptor("color_Color")
	node : RampShader = None
	pass
class Color_ColorPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	color_ColorB_ : Color_ColorBPlug = PlugDescriptor("color_ColorB")
	clrcb_ : Color_ColorBPlug = PlugDescriptor("color_ColorB")
	color_ColorG_ : Color_ColorGPlug = PlugDescriptor("color_ColorG")
	clrcg_ : Color_ColorGPlug = PlugDescriptor("color_ColorG")
	color_ColorR_ : Color_ColorRPlug = PlugDescriptor("color_ColorR")
	clrcr_ : Color_ColorRPlug = PlugDescriptor("color_ColorR")
	node : RampShader = None
	pass
class Color_InterpPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : RampShader = None
	pass
class Color_PositionPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : RampShader = None
	pass
class ColorPlug(Plug):
	color_Color_ : Color_ColorPlug = PlugDescriptor("color_Color")
	clrc_ : Color_ColorPlug = PlugDescriptor("color_Color")
	color_Interp_ : Color_InterpPlug = PlugDescriptor("color_Interp")
	clri_ : Color_InterpPlug = PlugDescriptor("color_Interp")
	color_Position_ : Color_PositionPlug = PlugDescriptor("color_Position")
	clrp_ : Color_PositionPlug = PlugDescriptor("color_Position")
	node : RampShader = None
	pass
class ColorInputPlug(Plug):
	node : RampShader = None
	pass
class DiffusePlug(Plug):
	node : RampShader = None
	pass
class EccentricityPlug(Plug):
	node : RampShader = None
	pass
class Environment_ColorBPlug(Plug):
	parent : Environment_ColorPlug = PlugDescriptor("environment_Color")
	node : RampShader = None
	pass
class Environment_ColorGPlug(Plug):
	parent : Environment_ColorPlug = PlugDescriptor("environment_Color")
	node : RampShader = None
	pass
class Environment_ColorRPlug(Plug):
	parent : Environment_ColorPlug = PlugDescriptor("environment_Color")
	node : RampShader = None
	pass
class Environment_ColorPlug(Plug):
	parent : EnvironmentPlug = PlugDescriptor("environment")
	environment_ColorB_ : Environment_ColorBPlug = PlugDescriptor("environment_ColorB")
	envcb_ : Environment_ColorBPlug = PlugDescriptor("environment_ColorB")
	environment_ColorG_ : Environment_ColorGPlug = PlugDescriptor("environment_ColorG")
	envcg_ : Environment_ColorGPlug = PlugDescriptor("environment_ColorG")
	environment_ColorR_ : Environment_ColorRPlug = PlugDescriptor("environment_ColorR")
	envcr_ : Environment_ColorRPlug = PlugDescriptor("environment_ColorR")
	node : RampShader = None
	pass
class Environment_InterpPlug(Plug):
	parent : EnvironmentPlug = PlugDescriptor("environment")
	node : RampShader = None
	pass
class Environment_PositionPlug(Plug):
	parent : EnvironmentPlug = PlugDescriptor("environment")
	node : RampShader = None
	pass
class EnvironmentPlug(Plug):
	environment_Color_ : Environment_ColorPlug = PlugDescriptor("environment_Color")
	envc_ : Environment_ColorPlug = PlugDescriptor("environment_Color")
	environment_Interp_ : Environment_InterpPlug = PlugDescriptor("environment_Interp")
	envi_ : Environment_InterpPlug = PlugDescriptor("environment_Interp")
	environment_Position_ : Environment_PositionPlug = PlugDescriptor("environment_Position")
	envp_ : Environment_PositionPlug = PlugDescriptor("environment_Position")
	node : RampShader = None
	pass
class ForwardScatterPlug(Plug):
	node : RampShader = None
	pass
class GlowIntensityPlug(Plug):
	node : RampShader = None
	pass
class HideSourcePlug(Plug):
	node : RampShader = None
	pass
class Incandescence_ColorBPlug(Plug):
	parent : Incandescence_ColorPlug = PlugDescriptor("incandescence_Color")
	node : RampShader = None
	pass
class Incandescence_ColorGPlug(Plug):
	parent : Incandescence_ColorPlug = PlugDescriptor("incandescence_Color")
	node : RampShader = None
	pass
class Incandescence_ColorRPlug(Plug):
	parent : Incandescence_ColorPlug = PlugDescriptor("incandescence_Color")
	node : RampShader = None
	pass
class Incandescence_ColorPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	incandescence_ColorB_ : Incandescence_ColorBPlug = PlugDescriptor("incandescence_ColorB")
	iccb_ : Incandescence_ColorBPlug = PlugDescriptor("incandescence_ColorB")
	incandescence_ColorG_ : Incandescence_ColorGPlug = PlugDescriptor("incandescence_ColorG")
	iccg_ : Incandescence_ColorGPlug = PlugDescriptor("incandescence_ColorG")
	incandescence_ColorR_ : Incandescence_ColorRPlug = PlugDescriptor("incandescence_ColorR")
	iccr_ : Incandescence_ColorRPlug = PlugDescriptor("incandescence_ColorR")
	node : RampShader = None
	pass
class Incandescence_InterpPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : RampShader = None
	pass
class Incandescence_PositionPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : RampShader = None
	pass
class IncandescencePlug(Plug):
	incandescence_Color_ : Incandescence_ColorPlug = PlugDescriptor("incandescence_Color")
	icc_ : Incandescence_ColorPlug = PlugDescriptor("incandescence_Color")
	incandescence_Interp_ : Incandescence_InterpPlug = PlugDescriptor("incandescence_Interp")
	ici_ : Incandescence_InterpPlug = PlugDescriptor("incandescence_Interp")
	incandescence_Position_ : Incandescence_PositionPlug = PlugDescriptor("incandescence_Position")
	icp_ : Incandescence_PositionPlug = PlugDescriptor("incandescence_Position")
	node : RampShader = None
	pass
class LightAbsorbancePlug(Plug):
	node : RampShader = None
	pass
class LightAmbientPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : RampShader = None
	pass
class LightBlindDataPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : RampShader = None
	pass
class LightDiffusePlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : RampShader = None
	pass
class LightDirectionXPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : RampShader = None
	pass
class LightDirectionYPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : RampShader = None
	pass
class LightDirectionZPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : RampShader = None
	pass
class LightDirectionPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	lightDirectionX_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	ldx_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	lightDirectionY_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	ldy_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	lightDirectionZ_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	ldz_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	node : RampShader = None
	pass
class LightIntensityBPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : RampShader = None
	pass
class LightIntensityGPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : RampShader = None
	pass
class LightIntensityRPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : RampShader = None
	pass
class LightIntensityPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	lightIntensityB_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lib_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lightIntensityG_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lig_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lightIntensityR_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	lir_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	node : RampShader = None
	pass
class LightShadowFractionPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : RampShader = None
	pass
class LightSpecularPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : RampShader = None
	pass
class PreShadowIntensityPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : RampShader = None
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
	node : RampShader = None
	pass
class MatrixEyeToWorldPlug(Plug):
	node : RampShader = None
	pass
class MatteOpacityPlug(Plug):
	node : RampShader = None
	pass
class MatteOpacityModePlug(Plug):
	node : RampShader = None
	pass
class MediumRefractiveIndexPlug(Plug):
	node : RampShader = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : RampShader = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : RampShader = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : RampShader = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	nx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ny_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	nz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : RampShader = None
	pass
class ObjectIdPlug(Plug):
	node : RampShader = None
	pass
class OpacityDepthPlug(Plug):
	node : RampShader = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : RampShader = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : RampShader = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : RampShader = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : RampShader = None
	pass
class OutGlowColorBPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : RampShader = None
	pass
class OutGlowColorGPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : RampShader = None
	pass
class OutGlowColorRPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : RampShader = None
	pass
class OutGlowColorPlug(Plug):
	outGlowColorB_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	ogb_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	outGlowColorG_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	ogg_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	outGlowColorR_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	ogr_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	node : RampShader = None
	pass
class OutMatteOpacityBPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : RampShader = None
	pass
class OutMatteOpacityGPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : RampShader = None
	pass
class OutMatteOpacityRPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : RampShader = None
	pass
class OutMatteOpacityPlug(Plug):
	outMatteOpacityB_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	omob_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	outMatteOpacityG_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	omog_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	outMatteOpacityR_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	omor_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	node : RampShader = None
	pass
class OutTransparencyBPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : RampShader = None
	pass
class OutTransparencyGPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : RampShader = None
	pass
class OutTransparencyRPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : RampShader = None
	pass
class OutTransparencyPlug(Plug):
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	otb_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	otg_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	otr_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	node : RampShader = None
	pass
class PointCameraXPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : RampShader = None
	pass
class PointCameraYPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : RampShader = None
	pass
class PointCameraZPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : RampShader = None
	pass
class PointCameraPlug(Plug):
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	px_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	py_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pz_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	node : RampShader = None
	pass
class PrimitiveIdPlug(Plug):
	node : RampShader = None
	pass
class RayDepthPlug(Plug):
	node : RampShader = None
	pass
class RayDirectionXPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : RampShader = None
	pass
class RayDirectionYPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : RampShader = None
	pass
class RayDirectionZPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : RampShader = None
	pass
class RayDirectionPlug(Plug):
	rayDirectionX_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rdx_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rayDirectionY_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rdy_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rayDirectionZ_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	rdz_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	node : RampShader = None
	pass
class RayInstancePlug(Plug):
	node : RampShader = None
	pass
class RaySamplerPlug(Plug):
	node : RampShader = None
	pass
class ReflectedColorBPlug(Plug):
	parent : ReflectedColorPlug = PlugDescriptor("reflectedColor")
	node : RampShader = None
	pass
class ReflectedColorGPlug(Plug):
	parent : ReflectedColorPlug = PlugDescriptor("reflectedColor")
	node : RampShader = None
	pass
class ReflectedColorRPlug(Plug):
	parent : ReflectedColorPlug = PlugDescriptor("reflectedColor")
	node : RampShader = None
	pass
class ReflectedColorPlug(Plug):
	reflectedColorB_ : ReflectedColorBPlug = PlugDescriptor("reflectedColorB")
	rb_ : ReflectedColorBPlug = PlugDescriptor("reflectedColorB")
	reflectedColorG_ : ReflectedColorGPlug = PlugDescriptor("reflectedColorG")
	rg_ : ReflectedColorGPlug = PlugDescriptor("reflectedColorG")
	reflectedColorR_ : ReflectedColorRPlug = PlugDescriptor("reflectedColorR")
	rr_ : ReflectedColorRPlug = PlugDescriptor("reflectedColorR")
	node : RampShader = None
	pass
class ReflectionLimitPlug(Plug):
	node : RampShader = None
	pass
class ReflectionSpecularityPlug(Plug):
	node : RampShader = None
	pass
class Reflectivity_FloatValuePlug(Plug):
	parent : ReflectivityPlug = PlugDescriptor("reflectivity")
	node : RampShader = None
	pass
class Reflectivity_InterpPlug(Plug):
	parent : ReflectivityPlug = PlugDescriptor("reflectivity")
	node : RampShader = None
	pass
class Reflectivity_PositionPlug(Plug):
	parent : ReflectivityPlug = PlugDescriptor("reflectivity")
	node : RampShader = None
	pass
class ReflectivityPlug(Plug):
	reflectivity_FloatValue_ : Reflectivity_FloatValuePlug = PlugDescriptor("reflectivity_FloatValue")
	rflfv_ : Reflectivity_FloatValuePlug = PlugDescriptor("reflectivity_FloatValue")
	reflectivity_Interp_ : Reflectivity_InterpPlug = PlugDescriptor("reflectivity_Interp")
	rfli_ : Reflectivity_InterpPlug = PlugDescriptor("reflectivity_Interp")
	reflectivity_Position_ : Reflectivity_PositionPlug = PlugDescriptor("reflectivity_Position")
	rflp_ : Reflectivity_PositionPlug = PlugDescriptor("reflectivity_Position")
	node : RampShader = None
	pass
class RefractionLimitPlug(Plug):
	node : RampShader = None
	pass
class RefractionsPlug(Plug):
	node : RampShader = None
	pass
class RefractiveIndexPlug(Plug):
	node : RampShader = None
	pass
class ShadowAttenuationPlug(Plug):
	node : RampShader = None
	pass
class ShadowColorBPlug(Plug):
	parent : ShadowColorPlug = PlugDescriptor("shadowColor")
	node : RampShader = None
	pass
class ShadowColorGPlug(Plug):
	parent : ShadowColorPlug = PlugDescriptor("shadowColor")
	node : RampShader = None
	pass
class ShadowColorRPlug(Plug):
	parent : ShadowColorPlug = PlugDescriptor("shadowColor")
	node : RampShader = None
	pass
class ShadowColorPlug(Plug):
	shadowColorB_ : ShadowColorBPlug = PlugDescriptor("shadowColorB")
	shb_ : ShadowColorBPlug = PlugDescriptor("shadowColorB")
	shadowColorG_ : ShadowColorGPlug = PlugDescriptor("shadowColorG")
	shg_ : ShadowColorGPlug = PlugDescriptor("shadowColorG")
	shadowColorR_ : ShadowColorRPlug = PlugDescriptor("shadowColorR")
	shr_ : ShadowColorRPlug = PlugDescriptor("shadowColorR")
	node : RampShader = None
	pass
class ShadowModePlug(Plug):
	node : RampShader = None
	pass
class ShadowThresholdPlug(Plug):
	node : RampShader = None
	pass
class SpecularColor_ColorBPlug(Plug):
	parent : SpecularColor_ColorPlug = PlugDescriptor("specularColor_Color")
	node : RampShader = None
	pass
class SpecularColor_ColorGPlug(Plug):
	parent : SpecularColor_ColorPlug = PlugDescriptor("specularColor_Color")
	node : RampShader = None
	pass
class SpecularColor_ColorRPlug(Plug):
	parent : SpecularColor_ColorPlug = PlugDescriptor("specularColor_Color")
	node : RampShader = None
	pass
class SpecularColor_ColorPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	specularColor_ColorB_ : SpecularColor_ColorBPlug = PlugDescriptor("specularColor_ColorB")
	sccb_ : SpecularColor_ColorBPlug = PlugDescriptor("specularColor_ColorB")
	specularColor_ColorG_ : SpecularColor_ColorGPlug = PlugDescriptor("specularColor_ColorG")
	sccg_ : SpecularColor_ColorGPlug = PlugDescriptor("specularColor_ColorG")
	specularColor_ColorR_ : SpecularColor_ColorRPlug = PlugDescriptor("specularColor_ColorR")
	sccr_ : SpecularColor_ColorRPlug = PlugDescriptor("specularColor_ColorR")
	node : RampShader = None
	pass
class SpecularColor_InterpPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : RampShader = None
	pass
class SpecularColor_PositionPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : RampShader = None
	pass
class SpecularColorPlug(Plug):
	specularColor_Color_ : SpecularColor_ColorPlug = PlugDescriptor("specularColor_Color")
	scc_ : SpecularColor_ColorPlug = PlugDescriptor("specularColor_Color")
	specularColor_Interp_ : SpecularColor_InterpPlug = PlugDescriptor("specularColor_Interp")
	sci_ : SpecularColor_InterpPlug = PlugDescriptor("specularColor_Interp")
	specularColor_Position_ : SpecularColor_PositionPlug = PlugDescriptor("specularColor_Position")
	scp_ : SpecularColor_PositionPlug = PlugDescriptor("specularColor_Position")
	node : RampShader = None
	pass
class SpecularGlowPlug(Plug):
	node : RampShader = None
	pass
class SpecularRollOff_FloatValuePlug(Plug):
	parent : SpecularRollOffPlug = PlugDescriptor("specularRollOff")
	node : RampShader = None
	pass
class SpecularRollOff_InterpPlug(Plug):
	parent : SpecularRollOffPlug = PlugDescriptor("specularRollOff")
	node : RampShader = None
	pass
class SpecularRollOff_PositionPlug(Plug):
	parent : SpecularRollOffPlug = PlugDescriptor("specularRollOff")
	node : RampShader = None
	pass
class SpecularRollOffPlug(Plug):
	specularRollOff_FloatValue_ : SpecularRollOff_FloatValuePlug = PlugDescriptor("specularRollOff_FloatValue")
	srofv_ : SpecularRollOff_FloatValuePlug = PlugDescriptor("specularRollOff_FloatValue")
	specularRollOff_Interp_ : SpecularRollOff_InterpPlug = PlugDescriptor("specularRollOff_Interp")
	sroi_ : SpecularRollOff_InterpPlug = PlugDescriptor("specularRollOff_Interp")
	specularRollOff_Position_ : SpecularRollOff_PositionPlug = PlugDescriptor("specularRollOff_Position")
	srop_ : SpecularRollOff_PositionPlug = PlugDescriptor("specularRollOff_Position")
	node : RampShader = None
	pass
class SpecularityPlug(Plug):
	node : RampShader = None
	pass
class SurfaceThicknessPlug(Plug):
	node : RampShader = None
	pass
class TranslucencePlug(Plug):
	node : RampShader = None
	pass
class TranslucenceDepthPlug(Plug):
	node : RampShader = None
	pass
class TranslucenceFocusPlug(Plug):
	node : RampShader = None
	pass
class Transparency_ColorBPlug(Plug):
	parent : Transparency_ColorPlug = PlugDescriptor("transparency_Color")
	node : RampShader = None
	pass
class Transparency_ColorGPlug(Plug):
	parent : Transparency_ColorPlug = PlugDescriptor("transparency_Color")
	node : RampShader = None
	pass
class Transparency_ColorRPlug(Plug):
	parent : Transparency_ColorPlug = PlugDescriptor("transparency_Color")
	node : RampShader = None
	pass
class Transparency_ColorPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	transparency_ColorB_ : Transparency_ColorBPlug = PlugDescriptor("transparency_ColorB")
	itcb_ : Transparency_ColorBPlug = PlugDescriptor("transparency_ColorB")
	transparency_ColorG_ : Transparency_ColorGPlug = PlugDescriptor("transparency_ColorG")
	itcg_ : Transparency_ColorGPlug = PlugDescriptor("transparency_ColorG")
	transparency_ColorR_ : Transparency_ColorRPlug = PlugDescriptor("transparency_ColorR")
	itcr_ : Transparency_ColorRPlug = PlugDescriptor("transparency_ColorR")
	node : RampShader = None
	pass
class Transparency_InterpPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : RampShader = None
	pass
class Transparency_PositionPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : RampShader = None
	pass
class TransparencyPlug(Plug):
	transparency_Color_ : Transparency_ColorPlug = PlugDescriptor("transparency_Color")
	itc_ : Transparency_ColorPlug = PlugDescriptor("transparency_Color")
	transparency_Interp_ : Transparency_InterpPlug = PlugDescriptor("transparency_Interp")
	iti_ : Transparency_InterpPlug = PlugDescriptor("transparency_Interp")
	transparency_Position_ : Transparency_PositionPlug = PlugDescriptor("transparency_Position")
	itp_ : Transparency_PositionPlug = PlugDescriptor("transparency_Position")
	node : RampShader = None
	pass
class TransparencyDepthPlug(Plug):
	node : RampShader = None
	pass
class TriangleNormalCameraXPlug(Plug):
	parent : TriangleNormalCameraPlug = PlugDescriptor("triangleNormalCamera")
	node : RampShader = None
	pass
class TriangleNormalCameraYPlug(Plug):
	parent : TriangleNormalCameraPlug = PlugDescriptor("triangleNormalCamera")
	node : RampShader = None
	pass
class TriangleNormalCameraZPlug(Plug):
	parent : TriangleNormalCameraPlug = PlugDescriptor("triangleNormalCamera")
	node : RampShader = None
	pass
class TriangleNormalCameraPlug(Plug):
	triangleNormalCameraX_ : TriangleNormalCameraXPlug = PlugDescriptor("triangleNormalCameraX")
	tnx_ : TriangleNormalCameraXPlug = PlugDescriptor("triangleNormalCameraX")
	triangleNormalCameraY_ : TriangleNormalCameraYPlug = PlugDescriptor("triangleNormalCameraY")
	tny_ : TriangleNormalCameraYPlug = PlugDescriptor("triangleNormalCameraY")
	triangleNormalCameraZ_ : TriangleNormalCameraZPlug = PlugDescriptor("triangleNormalCameraZ")
	tnz_ : TriangleNormalCameraZPlug = PlugDescriptor("triangleNormalCameraZ")
	node : RampShader = None
	pass
# endregion


# define node class
class RampShader(ShadingDependNode):
	ambientColorB_ : AmbientColorBPlug = PlugDescriptor("ambientColorB")
	ambientColorG_ : AmbientColorGPlug = PlugDescriptor("ambientColorG")
	ambientColorR_ : AmbientColorRPlug = PlugDescriptor("ambientColorR")
	ambientColor_ : AmbientColorPlug = PlugDescriptor("ambientColor")
	chromaticAberration_ : ChromaticAberrationPlug = PlugDescriptor("chromaticAberration")
	color_ColorB_ : Color_ColorBPlug = PlugDescriptor("color_ColorB")
	color_ColorG_ : Color_ColorGPlug = PlugDescriptor("color_ColorG")
	color_ColorR_ : Color_ColorRPlug = PlugDescriptor("color_ColorR")
	color_Color_ : Color_ColorPlug = PlugDescriptor("color_Color")
	color_Interp_ : Color_InterpPlug = PlugDescriptor("color_Interp")
	color_Position_ : Color_PositionPlug = PlugDescriptor("color_Position")
	color_ : ColorPlug = PlugDescriptor("color")
	colorInput_ : ColorInputPlug = PlugDescriptor("colorInput")
	diffuse_ : DiffusePlug = PlugDescriptor("diffuse")
	eccentricity_ : EccentricityPlug = PlugDescriptor("eccentricity")
	environment_ColorB_ : Environment_ColorBPlug = PlugDescriptor("environment_ColorB")
	environment_ColorG_ : Environment_ColorGPlug = PlugDescriptor("environment_ColorG")
	environment_ColorR_ : Environment_ColorRPlug = PlugDescriptor("environment_ColorR")
	environment_Color_ : Environment_ColorPlug = PlugDescriptor("environment_Color")
	environment_Interp_ : Environment_InterpPlug = PlugDescriptor("environment_Interp")
	environment_Position_ : Environment_PositionPlug = PlugDescriptor("environment_Position")
	environment_ : EnvironmentPlug = PlugDescriptor("environment")
	forwardScatter_ : ForwardScatterPlug = PlugDescriptor("forwardScatter")
	glowIntensity_ : GlowIntensityPlug = PlugDescriptor("glowIntensity")
	hideSource_ : HideSourcePlug = PlugDescriptor("hideSource")
	incandescence_ColorB_ : Incandescence_ColorBPlug = PlugDescriptor("incandescence_ColorB")
	incandescence_ColorG_ : Incandescence_ColorGPlug = PlugDescriptor("incandescence_ColorG")
	incandescence_ColorR_ : Incandescence_ColorRPlug = PlugDescriptor("incandescence_ColorR")
	incandescence_Color_ : Incandescence_ColorPlug = PlugDescriptor("incandescence_Color")
	incandescence_Interp_ : Incandescence_InterpPlug = PlugDescriptor("incandescence_Interp")
	incandescence_Position_ : Incandescence_PositionPlug = PlugDescriptor("incandescence_Position")
	incandescence_ : IncandescencePlug = PlugDescriptor("incandescence")
	lightAbsorbance_ : LightAbsorbancePlug = PlugDescriptor("lightAbsorbance")
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
	matteOpacity_ : MatteOpacityPlug = PlugDescriptor("matteOpacity")
	matteOpacityMode_ : MatteOpacityModePlug = PlugDescriptor("matteOpacityMode")
	mediumRefractiveIndex_ : MediumRefractiveIndexPlug = PlugDescriptor("mediumRefractiveIndex")
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	normalCamera_ : NormalCameraPlug = PlugDescriptor("normalCamera")
	objectId_ : ObjectIdPlug = PlugDescriptor("objectId")
	opacityDepth_ : OpacityDepthPlug = PlugDescriptor("opacityDepth")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
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
	reflectedColorB_ : ReflectedColorBPlug = PlugDescriptor("reflectedColorB")
	reflectedColorG_ : ReflectedColorGPlug = PlugDescriptor("reflectedColorG")
	reflectedColorR_ : ReflectedColorRPlug = PlugDescriptor("reflectedColorR")
	reflectedColor_ : ReflectedColorPlug = PlugDescriptor("reflectedColor")
	reflectionLimit_ : ReflectionLimitPlug = PlugDescriptor("reflectionLimit")
	reflectionSpecularity_ : ReflectionSpecularityPlug = PlugDescriptor("reflectionSpecularity")
	reflectivity_FloatValue_ : Reflectivity_FloatValuePlug = PlugDescriptor("reflectivity_FloatValue")
	reflectivity_Interp_ : Reflectivity_InterpPlug = PlugDescriptor("reflectivity_Interp")
	reflectivity_Position_ : Reflectivity_PositionPlug = PlugDescriptor("reflectivity_Position")
	reflectivity_ : ReflectivityPlug = PlugDescriptor("reflectivity")
	refractionLimit_ : RefractionLimitPlug = PlugDescriptor("refractionLimit")
	refractions_ : RefractionsPlug = PlugDescriptor("refractions")
	refractiveIndex_ : RefractiveIndexPlug = PlugDescriptor("refractiveIndex")
	shadowAttenuation_ : ShadowAttenuationPlug = PlugDescriptor("shadowAttenuation")
	shadowColorB_ : ShadowColorBPlug = PlugDescriptor("shadowColorB")
	shadowColorG_ : ShadowColorGPlug = PlugDescriptor("shadowColorG")
	shadowColorR_ : ShadowColorRPlug = PlugDescriptor("shadowColorR")
	shadowColor_ : ShadowColorPlug = PlugDescriptor("shadowColor")
	shadowMode_ : ShadowModePlug = PlugDescriptor("shadowMode")
	shadowThreshold_ : ShadowThresholdPlug = PlugDescriptor("shadowThreshold")
	specularColor_ColorB_ : SpecularColor_ColorBPlug = PlugDescriptor("specularColor_ColorB")
	specularColor_ColorG_ : SpecularColor_ColorGPlug = PlugDescriptor("specularColor_ColorG")
	specularColor_ColorR_ : SpecularColor_ColorRPlug = PlugDescriptor("specularColor_ColorR")
	specularColor_Color_ : SpecularColor_ColorPlug = PlugDescriptor("specularColor_Color")
	specularColor_Interp_ : SpecularColor_InterpPlug = PlugDescriptor("specularColor_Interp")
	specularColor_Position_ : SpecularColor_PositionPlug = PlugDescriptor("specularColor_Position")
	specularColor_ : SpecularColorPlug = PlugDescriptor("specularColor")
	specularGlow_ : SpecularGlowPlug = PlugDescriptor("specularGlow")
	specularRollOff_FloatValue_ : SpecularRollOff_FloatValuePlug = PlugDescriptor("specularRollOff_FloatValue")
	specularRollOff_Interp_ : SpecularRollOff_InterpPlug = PlugDescriptor("specularRollOff_Interp")
	specularRollOff_Position_ : SpecularRollOff_PositionPlug = PlugDescriptor("specularRollOff_Position")
	specularRollOff_ : SpecularRollOffPlug = PlugDescriptor("specularRollOff")
	specularity_ : SpecularityPlug = PlugDescriptor("specularity")
	surfaceThickness_ : SurfaceThicknessPlug = PlugDescriptor("surfaceThickness")
	translucence_ : TranslucencePlug = PlugDescriptor("translucence")
	translucenceDepth_ : TranslucenceDepthPlug = PlugDescriptor("translucenceDepth")
	translucenceFocus_ : TranslucenceFocusPlug = PlugDescriptor("translucenceFocus")
	transparency_ColorB_ : Transparency_ColorBPlug = PlugDescriptor("transparency_ColorB")
	transparency_ColorG_ : Transparency_ColorGPlug = PlugDescriptor("transparency_ColorG")
	transparency_ColorR_ : Transparency_ColorRPlug = PlugDescriptor("transparency_ColorR")
	transparency_Color_ : Transparency_ColorPlug = PlugDescriptor("transparency_Color")
	transparency_Interp_ : Transparency_InterpPlug = PlugDescriptor("transparency_Interp")
	transparency_Position_ : Transparency_PositionPlug = PlugDescriptor("transparency_Position")
	transparency_ : TransparencyPlug = PlugDescriptor("transparency")
	transparencyDepth_ : TransparencyDepthPlug = PlugDescriptor("transparencyDepth")
	triangleNormalCameraX_ : TriangleNormalCameraXPlug = PlugDescriptor("triangleNormalCameraX")
	triangleNormalCameraY_ : TriangleNormalCameraYPlug = PlugDescriptor("triangleNormalCameraY")
	triangleNormalCameraZ_ : TriangleNormalCameraZPlug = PlugDescriptor("triangleNormalCameraZ")
	triangleNormalCamera_ : TriangleNormalCameraPlug = PlugDescriptor("triangleNormalCamera")

	# node attributes

	typeName = "rampShader"
	apiTypeInt = 896
	apiTypeStr = "kRampShader"
	typeIdInt = 1381126227
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["ambientColorB", "ambientColorG", "ambientColorR", "ambientColor", "chromaticAberration", "color_ColorB", "color_ColorG", "color_ColorR", "color_Color", "color_Interp", "color_Position", "color", "colorInput", "diffuse", "eccentricity", "environment_ColorB", "environment_ColorG", "environment_ColorR", "environment_Color", "environment_Interp", "environment_Position", "environment", "forwardScatter", "glowIntensity", "hideSource", "incandescence_ColorB", "incandescence_ColorG", "incandescence_ColorR", "incandescence_Color", "incandescence_Interp", "incandescence_Position", "incandescence", "lightAbsorbance", "lightAmbient", "lightBlindData", "lightDiffuse", "lightDirectionX", "lightDirectionY", "lightDirectionZ", "lightDirection", "lightIntensityB", "lightIntensityG", "lightIntensityR", "lightIntensity", "lightShadowFraction", "lightSpecular", "preShadowIntensity", "lightDataArray", "matrixEyeToWorld", "matteOpacity", "matteOpacityMode", "mediumRefractiveIndex", "normalCameraX", "normalCameraY", "normalCameraZ", "normalCamera", "objectId", "opacityDepth", "outColorB", "outColorG", "outColorR", "outColor", "outGlowColorB", "outGlowColorG", "outGlowColorR", "outGlowColor", "outMatteOpacityB", "outMatteOpacityG", "outMatteOpacityR", "outMatteOpacity", "outTransparencyB", "outTransparencyG", "outTransparencyR", "outTransparency", "pointCameraX", "pointCameraY", "pointCameraZ", "pointCamera", "primitiveId", "rayDepth", "rayDirectionX", "rayDirectionY", "rayDirectionZ", "rayDirection", "rayInstance", "raySampler", "reflectedColorB", "reflectedColorG", "reflectedColorR", "reflectedColor", "reflectionLimit", "reflectionSpecularity", "reflectivity_FloatValue", "reflectivity_Interp", "reflectivity_Position", "reflectivity", "refractionLimit", "refractions", "refractiveIndex", "shadowAttenuation", "shadowColorB", "shadowColorG", "shadowColorR", "shadowColor", "shadowMode", "shadowThreshold", "specularColor_ColorB", "specularColor_ColorG", "specularColor_ColorR", "specularColor_Color", "specularColor_Interp", "specularColor_Position", "specularColor", "specularGlow", "specularRollOff_FloatValue", "specularRollOff_Interp", "specularRollOff_Position", "specularRollOff", "specularity", "surfaceThickness", "translucence", "translucenceDepth", "translucenceFocus", "transparency_ColorB", "transparency_ColorG", "transparency_ColorR", "transparency_Color", "transparency_Interp", "transparency_Position", "transparency", "transparencyDepth", "triangleNormalCameraX", "triangleNormalCameraY", "triangleNormalCameraZ", "triangleNormalCamera"]
	nodeLeafPlugs = ["ambientColor", "chromaticAberration", "color", "colorInput", "diffuse", "eccentricity", "environment", "forwardScatter", "glowIntensity", "hideSource", "incandescence", "lightAbsorbance", "lightDataArray", "matrixEyeToWorld", "matteOpacity", "matteOpacityMode", "mediumRefractiveIndex", "normalCamera", "objectId", "opacityDepth", "outColor", "outGlowColor", "outMatteOpacity", "outTransparency", "pointCamera", "primitiveId", "rayDepth", "rayDirection", "rayInstance", "raySampler", "reflectedColor", "reflectionLimit", "reflectionSpecularity", "reflectivity", "refractionLimit", "refractions", "refractiveIndex", "shadowAttenuation", "shadowColor", "shadowMode", "shadowThreshold", "specularColor", "specularGlow", "specularRollOff", "specularity", "surfaceThickness", "translucence", "translucenceDepth", "translucenceFocus", "transparency", "transparencyDepth", "triangleNormalCamera"]
	pass

