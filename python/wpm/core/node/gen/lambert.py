

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PaintableShadingDependNode = retriever.getNodeCls("PaintableShadingDependNode")
assert PaintableShadingDependNode
if T.TYPE_CHECKING:
	from .. import PaintableShadingDependNode

# add node doc



# region plug type defs
class AmbientColorBPlug(Plug):
	parent : AmbientColorPlug = PlugDescriptor("ambientColor")
	node : Lambert = None
	pass
class AmbientColorGPlug(Plug):
	parent : AmbientColorPlug = PlugDescriptor("ambientColor")
	node : Lambert = None
	pass
class AmbientColorRPlug(Plug):
	parent : AmbientColorPlug = PlugDescriptor("ambientColor")
	node : Lambert = None
	pass
class AmbientColorPlug(Plug):
	ambientColorB_ : AmbientColorBPlug = PlugDescriptor("ambientColorB")
	acb_ : AmbientColorBPlug = PlugDescriptor("ambientColorB")
	ambientColorG_ : AmbientColorGPlug = PlugDescriptor("ambientColorG")
	acg_ : AmbientColorGPlug = PlugDescriptor("ambientColorG")
	ambientColorR_ : AmbientColorRPlug = PlugDescriptor("ambientColorR")
	acr_ : AmbientColorRPlug = PlugDescriptor("ambientColorR")
	node : Lambert = None
	pass
class ChromaticAberrationPlug(Plug):
	node : Lambert = None
	pass
class ColorBPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : Lambert = None
	pass
class ColorGPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : Lambert = None
	pass
class ColorRPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : Lambert = None
	pass
class ColorPlug(Plug):
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	cb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	cg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	cr_ : ColorRPlug = PlugDescriptor("colorR")
	node : Lambert = None
	pass
class DiffusePlug(Plug):
	node : Lambert = None
	pass
class GlowIntensityPlug(Plug):
	node : Lambert = None
	pass
class HardwareShaderBPlug(Plug):
	parent : HardwareShaderPlug = PlugDescriptor("hardwareShader")
	node : Lambert = None
	pass
class HardwareShaderGPlug(Plug):
	parent : HardwareShaderPlug = PlugDescriptor("hardwareShader")
	node : Lambert = None
	pass
class HardwareShaderRPlug(Plug):
	parent : HardwareShaderPlug = PlugDescriptor("hardwareShader")
	node : Lambert = None
	pass
class HardwareShaderPlug(Plug):
	hardwareShaderB_ : HardwareShaderBPlug = PlugDescriptor("hardwareShaderB")
	hwb_ : HardwareShaderBPlug = PlugDescriptor("hardwareShaderB")
	hardwareShaderG_ : HardwareShaderGPlug = PlugDescriptor("hardwareShaderG")
	hwg_ : HardwareShaderGPlug = PlugDescriptor("hardwareShaderG")
	hardwareShaderR_ : HardwareShaderRPlug = PlugDescriptor("hardwareShaderR")
	hwr_ : HardwareShaderRPlug = PlugDescriptor("hardwareShaderR")
	node : Lambert = None
	pass
class HideSourcePlug(Plug):
	node : Lambert = None
	pass
class IncandescenceBPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : Lambert = None
	pass
class IncandescenceGPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : Lambert = None
	pass
class IncandescenceRPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : Lambert = None
	pass
class IncandescencePlug(Plug):
	incandescenceB_ : IncandescenceBPlug = PlugDescriptor("incandescenceB")
	ib_ : IncandescenceBPlug = PlugDescriptor("incandescenceB")
	incandescenceG_ : IncandescenceGPlug = PlugDescriptor("incandescenceG")
	ig_ : IncandescenceGPlug = PlugDescriptor("incandescenceG")
	incandescenceR_ : IncandescenceRPlug = PlugDescriptor("incandescenceR")
	ir_ : IncandescenceRPlug = PlugDescriptor("incandescenceR")
	node : Lambert = None
	pass
class LightAbsorbancePlug(Plug):
	node : Lambert = None
	pass
class LightAmbientPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : Lambert = None
	pass
class LightBlindDataPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : Lambert = None
	pass
class LightDiffusePlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : Lambert = None
	pass
class LightDirectionXPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : Lambert = None
	pass
class LightDirectionYPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : Lambert = None
	pass
class LightDirectionZPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : Lambert = None
	pass
class LightDirectionPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	lightDirectionX_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	ldx_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	lightDirectionY_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	ldy_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	lightDirectionZ_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	ldz_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	node : Lambert = None
	pass
class LightIntensityBPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : Lambert = None
	pass
class LightIntensityGPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : Lambert = None
	pass
class LightIntensityRPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : Lambert = None
	pass
class LightIntensityPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	lightIntensityB_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lib_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lightIntensityG_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lig_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lightIntensityR_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	lir_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	node : Lambert = None
	pass
class LightShadowFractionPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : Lambert = None
	pass
class LightSpecularPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : Lambert = None
	pass
class PreShadowIntensityPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : Lambert = None
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
	node : Lambert = None
	pass
class MaterialAlphaGainPlug(Plug):
	node : Lambert = None
	pass
class MatteOpacityPlug(Plug):
	node : Lambert = None
	pass
class MatteOpacityModePlug(Plug):
	node : Lambert = None
	pass
class MediumRefractiveIndexPlug(Plug):
	node : Lambert = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Lambert = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Lambert = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : Lambert = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	nx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ny_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	nz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : Lambert = None
	pass
class ObjectIdPlug(Plug):
	node : Lambert = None
	pass
class OpacityDepthPlug(Plug):
	node : Lambert = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : Lambert = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : Lambert = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : Lambert = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : Lambert = None
	pass
class OutGlowColorBPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : Lambert = None
	pass
class OutGlowColorGPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : Lambert = None
	pass
class OutGlowColorRPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : Lambert = None
	pass
class OutGlowColorPlug(Plug):
	outGlowColorB_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	ogb_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	outGlowColorG_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	ogg_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	outGlowColorR_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	ogr_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	node : Lambert = None
	pass
class OutMatteOpacityBPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : Lambert = None
	pass
class OutMatteOpacityGPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : Lambert = None
	pass
class OutMatteOpacityRPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : Lambert = None
	pass
class OutMatteOpacityPlug(Plug):
	outMatteOpacityB_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	omob_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	outMatteOpacityG_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	omog_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	outMatteOpacityR_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	omor_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	node : Lambert = None
	pass
class OutTransparencyBPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : Lambert = None
	pass
class OutTransparencyGPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : Lambert = None
	pass
class OutTransparencyRPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : Lambert = None
	pass
class OutTransparencyPlug(Plug):
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	otb_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	otg_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	otr_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	node : Lambert = None
	pass
class PointCameraXPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : Lambert = None
	pass
class PointCameraYPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : Lambert = None
	pass
class PointCameraZPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : Lambert = None
	pass
class PointCameraPlug(Plug):
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	px_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	py_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pz_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	node : Lambert = None
	pass
class PrimitiveIdPlug(Plug):
	node : Lambert = None
	pass
class RayDepthPlug(Plug):
	node : Lambert = None
	pass
class RayDirectionXPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : Lambert = None
	pass
class RayDirectionYPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : Lambert = None
	pass
class RayDirectionZPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : Lambert = None
	pass
class RayDirectionPlug(Plug):
	rayDirectionX_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rdx_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rayDirectionY_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rdy_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rayDirectionZ_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	rdz_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	node : Lambert = None
	pass
class RayInstancePlug(Plug):
	node : Lambert = None
	pass
class RaySamplerPlug(Plug):
	node : Lambert = None
	pass
class RefractionLimitPlug(Plug):
	node : Lambert = None
	pass
class RefractionsPlug(Plug):
	node : Lambert = None
	pass
class RefractiveIndexPlug(Plug):
	node : Lambert = None
	pass
class ShadowAttenuationPlug(Plug):
	node : Lambert = None
	pass
class SurfaceThicknessPlug(Plug):
	node : Lambert = None
	pass
class TranslucencePlug(Plug):
	node : Lambert = None
	pass
class TranslucenceDepthPlug(Plug):
	node : Lambert = None
	pass
class TranslucenceFocusPlug(Plug):
	node : Lambert = None
	pass
class TransparencyBPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : Lambert = None
	pass
class TransparencyGPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : Lambert = None
	pass
class TransparencyRPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : Lambert = None
	pass
class TransparencyPlug(Plug):
	transparencyB_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	itb_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	transparencyG_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	itg_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	transparencyR_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	itr_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	node : Lambert = None
	pass
class TransparencyDepthPlug(Plug):
	node : Lambert = None
	pass
class VrEdgeColorBPlug(Plug):
	parent : VrEdgeColorPlug = PlugDescriptor("vrEdgeColor")
	node : Lambert = None
	pass
class VrEdgeColorGPlug(Plug):
	parent : VrEdgeColorPlug = PlugDescriptor("vrEdgeColor")
	node : Lambert = None
	pass
class VrEdgeColorRPlug(Plug):
	parent : VrEdgeColorPlug = PlugDescriptor("vrEdgeColor")
	node : Lambert = None
	pass
class VrEdgeColorPlug(Plug):
	vrEdgeColorB_ : VrEdgeColorBPlug = PlugDescriptor("vrEdgeColorB")
	vecb_ : VrEdgeColorBPlug = PlugDescriptor("vrEdgeColorB")
	vrEdgeColorG_ : VrEdgeColorGPlug = PlugDescriptor("vrEdgeColorG")
	vecg_ : VrEdgeColorGPlug = PlugDescriptor("vrEdgeColorG")
	vrEdgeColorR_ : VrEdgeColorRPlug = PlugDescriptor("vrEdgeColorR")
	vecr_ : VrEdgeColorRPlug = PlugDescriptor("vrEdgeColorR")
	node : Lambert = None
	pass
class VrEdgePriorityPlug(Plug):
	node : Lambert = None
	pass
class VrEdgeStylePlug(Plug):
	node : Lambert = None
	pass
class VrEdgeWeightPlug(Plug):
	node : Lambert = None
	pass
class VrFillObjectPlug(Plug):
	node : Lambert = None
	pass
class VrHiddenEdgesPlug(Plug):
	node : Lambert = None
	pass
class VrHiddenEdgesOnTransparentPlug(Plug):
	node : Lambert = None
	pass
class VrOutlinesAtIntersectionsPlug(Plug):
	node : Lambert = None
	pass
class VrOverwriteDefaultsPlug(Plug):
	node : Lambert = None
	pass
# endregion


# define node class
class Lambert(PaintableShadingDependNode):
	ambientColorB_ : AmbientColorBPlug = PlugDescriptor("ambientColorB")
	ambientColorG_ : AmbientColorGPlug = PlugDescriptor("ambientColorG")
	ambientColorR_ : AmbientColorRPlug = PlugDescriptor("ambientColorR")
	ambientColor_ : AmbientColorPlug = PlugDescriptor("ambientColor")
	chromaticAberration_ : ChromaticAberrationPlug = PlugDescriptor("chromaticAberration")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	color_ : ColorPlug = PlugDescriptor("color")
	diffuse_ : DiffusePlug = PlugDescriptor("diffuse")
	glowIntensity_ : GlowIntensityPlug = PlugDescriptor("glowIntensity")
	hardwareShaderB_ : HardwareShaderBPlug = PlugDescriptor("hardwareShaderB")
	hardwareShaderG_ : HardwareShaderGPlug = PlugDescriptor("hardwareShaderG")
	hardwareShaderR_ : HardwareShaderRPlug = PlugDescriptor("hardwareShaderR")
	hardwareShader_ : HardwareShaderPlug = PlugDescriptor("hardwareShader")
	hideSource_ : HideSourcePlug = PlugDescriptor("hideSource")
	incandescenceB_ : IncandescenceBPlug = PlugDescriptor("incandescenceB")
	incandescenceG_ : IncandescenceGPlug = PlugDescriptor("incandescenceG")
	incandescenceR_ : IncandescenceRPlug = PlugDescriptor("incandescenceR")
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
	materialAlphaGain_ : MaterialAlphaGainPlug = PlugDescriptor("materialAlphaGain")
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
	refractionLimit_ : RefractionLimitPlug = PlugDescriptor("refractionLimit")
	refractions_ : RefractionsPlug = PlugDescriptor("refractions")
	refractiveIndex_ : RefractiveIndexPlug = PlugDescriptor("refractiveIndex")
	shadowAttenuation_ : ShadowAttenuationPlug = PlugDescriptor("shadowAttenuation")
	surfaceThickness_ : SurfaceThicknessPlug = PlugDescriptor("surfaceThickness")
	translucence_ : TranslucencePlug = PlugDescriptor("translucence")
	translucenceDepth_ : TranslucenceDepthPlug = PlugDescriptor("translucenceDepth")
	translucenceFocus_ : TranslucenceFocusPlug = PlugDescriptor("translucenceFocus")
	transparencyB_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	transparencyG_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	transparencyR_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	transparency_ : TransparencyPlug = PlugDescriptor("transparency")
	transparencyDepth_ : TransparencyDepthPlug = PlugDescriptor("transparencyDepth")
	vrEdgeColorB_ : VrEdgeColorBPlug = PlugDescriptor("vrEdgeColorB")
	vrEdgeColorG_ : VrEdgeColorGPlug = PlugDescriptor("vrEdgeColorG")
	vrEdgeColorR_ : VrEdgeColorRPlug = PlugDescriptor("vrEdgeColorR")
	vrEdgeColor_ : VrEdgeColorPlug = PlugDescriptor("vrEdgeColor")
	vrEdgePriority_ : VrEdgePriorityPlug = PlugDescriptor("vrEdgePriority")
	vrEdgeStyle_ : VrEdgeStylePlug = PlugDescriptor("vrEdgeStyle")
	vrEdgeWeight_ : VrEdgeWeightPlug = PlugDescriptor("vrEdgeWeight")
	vrFillObject_ : VrFillObjectPlug = PlugDescriptor("vrFillObject")
	vrHiddenEdges_ : VrHiddenEdgesPlug = PlugDescriptor("vrHiddenEdges")
	vrHiddenEdgesOnTransparent_ : VrHiddenEdgesOnTransparentPlug = PlugDescriptor("vrHiddenEdgesOnTransparent")
	vrOutlinesAtIntersections_ : VrOutlinesAtIntersectionsPlug = PlugDescriptor("vrOutlinesAtIntersections")
	vrOverwriteDefaults_ : VrOverwriteDefaultsPlug = PlugDescriptor("vrOverwriteDefaults")

	# node attributes

	typeName = "lambert"
	apiTypeInt = 371
	apiTypeStr = "kLambert"
	typeIdInt = 1380729165
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["ambientColorB", "ambientColorG", "ambientColorR", "ambientColor", "chromaticAberration", "colorB", "colorG", "colorR", "color", "diffuse", "glowIntensity", "hardwareShaderB", "hardwareShaderG", "hardwareShaderR", "hardwareShader", "hideSource", "incandescenceB", "incandescenceG", "incandescenceR", "incandescence", "lightAbsorbance", "lightAmbient", "lightBlindData", "lightDiffuse", "lightDirectionX", "lightDirectionY", "lightDirectionZ", "lightDirection", "lightIntensityB", "lightIntensityG", "lightIntensityR", "lightIntensity", "lightShadowFraction", "lightSpecular", "preShadowIntensity", "lightDataArray", "materialAlphaGain", "matteOpacity", "matteOpacityMode", "mediumRefractiveIndex", "normalCameraX", "normalCameraY", "normalCameraZ", "normalCamera", "objectId", "opacityDepth", "outColorB", "outColorG", "outColorR", "outColor", "outGlowColorB", "outGlowColorG", "outGlowColorR", "outGlowColor", "outMatteOpacityB", "outMatteOpacityG", "outMatteOpacityR", "outMatteOpacity", "outTransparencyB", "outTransparencyG", "outTransparencyR", "outTransparency", "pointCameraX", "pointCameraY", "pointCameraZ", "pointCamera", "primitiveId", "rayDepth", "rayDirectionX", "rayDirectionY", "rayDirectionZ", "rayDirection", "rayInstance", "raySampler", "refractionLimit", "refractions", "refractiveIndex", "shadowAttenuation", "surfaceThickness", "translucence", "translucenceDepth", "translucenceFocus", "transparencyB", "transparencyG", "transparencyR", "transparency", "transparencyDepth", "vrEdgeColorB", "vrEdgeColorG", "vrEdgeColorR", "vrEdgeColor", "vrEdgePriority", "vrEdgeStyle", "vrEdgeWeight", "vrFillObject", "vrHiddenEdges", "vrHiddenEdgesOnTransparent", "vrOutlinesAtIntersections", "vrOverwriteDefaults"]
	nodeLeafPlugs = ["ambientColor", "chromaticAberration", "color", "diffuse", "glowIntensity", "hardwareShader", "hideSource", "incandescence", "lightAbsorbance", "lightDataArray", "materialAlphaGain", "matteOpacity", "matteOpacityMode", "mediumRefractiveIndex", "normalCamera", "objectId", "opacityDepth", "outColor", "outGlowColor", "outMatteOpacity", "outTransparency", "pointCamera", "primitiveId", "rayDepth", "rayDirection", "rayInstance", "raySampler", "refractionLimit", "refractions", "refractiveIndex", "shadowAttenuation", "surfaceThickness", "translucence", "translucenceDepth", "translucenceFocus", "transparency", "transparencyDepth", "vrEdgeColor", "vrEdgePriority", "vrEdgeStyle", "vrEdgeWeight", "vrFillObject", "vrHiddenEdges", "vrHiddenEdgesOnTransparent", "vrOutlinesAtIntersections", "vrOverwriteDefaults"]
	pass

