

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
class AxialDropoffPlug(Plug):
	node : VolumeFog = None
	pass
class ColorBPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : VolumeFog = None
	pass
class ColorGPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : VolumeFog = None
	pass
class ColorRPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : VolumeFog = None
	pass
class ColorPlug(Plug):
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	cb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	cg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	cr_ : ColorRPlug = PlugDescriptor("colorR")
	node : VolumeFog = None
	pass
class ColorRamp_ColorBPlug(Plug):
	parent : ColorRamp_ColorPlug = PlugDescriptor("colorRamp_Color")
	node : VolumeFog = None
	pass
class ColorRamp_ColorGPlug(Plug):
	parent : ColorRamp_ColorPlug = PlugDescriptor("colorRamp_Color")
	node : VolumeFog = None
	pass
class ColorRamp_ColorRPlug(Plug):
	parent : ColorRamp_ColorPlug = PlugDescriptor("colorRamp_Color")
	node : VolumeFog = None
	pass
class ColorRamp_ColorPlug(Plug):
	parent : ColorRampPlug = PlugDescriptor("colorRamp")
	colorRamp_ColorB_ : ColorRamp_ColorBPlug = PlugDescriptor("colorRamp_ColorB")
	crmcb_ : ColorRamp_ColorBPlug = PlugDescriptor("colorRamp_ColorB")
	colorRamp_ColorG_ : ColorRamp_ColorGPlug = PlugDescriptor("colorRamp_ColorG")
	crmcg_ : ColorRamp_ColorGPlug = PlugDescriptor("colorRamp_ColorG")
	colorRamp_ColorR_ : ColorRamp_ColorRPlug = PlugDescriptor("colorRamp_ColorR")
	crmcr_ : ColorRamp_ColorRPlug = PlugDescriptor("colorRamp_ColorR")
	node : VolumeFog = None
	pass
class ColorRamp_InterpPlug(Plug):
	parent : ColorRampPlug = PlugDescriptor("colorRamp")
	node : VolumeFog = None
	pass
class ColorRamp_PositionPlug(Plug):
	parent : ColorRampPlug = PlugDescriptor("colorRamp")
	node : VolumeFog = None
	pass
class ColorRampPlug(Plug):
	colorRamp_Color_ : ColorRamp_ColorPlug = PlugDescriptor("colorRamp_Color")
	crmc_ : ColorRamp_ColorPlug = PlugDescriptor("colorRamp_Color")
	colorRamp_Interp_ : ColorRamp_InterpPlug = PlugDescriptor("colorRamp_Interp")
	crmi_ : ColorRamp_InterpPlug = PlugDescriptor("colorRamp_Interp")
	colorRamp_Position_ : ColorRamp_PositionPlug = PlugDescriptor("colorRamp_Position")
	crmp_ : ColorRamp_PositionPlug = PlugDescriptor("colorRamp_Position")
	node : VolumeFog = None
	pass
class ColorRampInputPlug(Plug):
	node : VolumeFog = None
	pass
class DensityPlug(Plug):
	node : VolumeFog = None
	pass
class DensityModePlug(Plug):
	node : VolumeFog = None
	pass
class DropoffMethodPlug(Plug):
	node : VolumeFog = None
	pass
class DropoffShapePlug(Plug):
	node : VolumeFog = None
	pass
class DropoffSubtractPlug(Plug):
	node : VolumeFog = None
	pass
class EdgeDropoffPlug(Plug):
	node : VolumeFog = None
	pass
class FarPointObjectXPlug(Plug):
	parent : FarPointObjPlug = PlugDescriptor("farPointObj")
	node : VolumeFog = None
	pass
class FarPointObjectYPlug(Plug):
	parent : FarPointObjPlug = PlugDescriptor("farPointObj")
	node : VolumeFog = None
	pass
class FarPointObjectZPlug(Plug):
	parent : FarPointObjPlug = PlugDescriptor("farPointObj")
	node : VolumeFog = None
	pass
class FarPointObjPlug(Plug):
	farPointObjectX_ : FarPointObjectXPlug = PlugDescriptor("farPointObjectX")
	fox_ : FarPointObjectXPlug = PlugDescriptor("farPointObjectX")
	farPointObjectY_ : FarPointObjectYPlug = PlugDescriptor("farPointObjectY")
	foy_ : FarPointObjectYPlug = PlugDescriptor("farPointObjectY")
	farPointObjectZ_ : FarPointObjectZPlug = PlugDescriptor("farPointObjectZ")
	foz_ : FarPointObjectZPlug = PlugDescriptor("farPointObjectZ")
	node : VolumeFog = None
	pass
class FarPointWorldXPlug(Plug):
	parent : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	node : VolumeFog = None
	pass
class FarPointWorldYPlug(Plug):
	parent : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	node : VolumeFog = None
	pass
class FarPointWorldZPlug(Plug):
	parent : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	node : VolumeFog = None
	pass
class FarPointWorldPlug(Plug):
	farPointWorldX_ : FarPointWorldXPlug = PlugDescriptor("farPointWorldX")
	fwx_ : FarPointWorldXPlug = PlugDescriptor("farPointWorldX")
	farPointWorldY_ : FarPointWorldYPlug = PlugDescriptor("farPointWorldY")
	fwy_ : FarPointWorldYPlug = PlugDescriptor("farPointWorldY")
	farPointWorldZ_ : FarPointWorldZPlug = PlugDescriptor("farPointWorldZ")
	fwz_ : FarPointWorldZPlug = PlugDescriptor("farPointWorldZ")
	node : VolumeFog = None
	pass
class GlowIntensityPlug(Plug):
	node : VolumeFog = None
	pass
class IlluminatedPlug(Plug):
	node : VolumeFog = None
	pass
class IncandescenceBPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : VolumeFog = None
	pass
class IncandescenceGPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : VolumeFog = None
	pass
class IncandescenceRPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : VolumeFog = None
	pass
class IncandescencePlug(Plug):
	incandescenceB_ : IncandescenceBPlug = PlugDescriptor("incandescenceB")
	ib_ : IncandescenceBPlug = PlugDescriptor("incandescenceB")
	incandescenceG_ : IncandescenceGPlug = PlugDescriptor("incandescenceG")
	ig_ : IncandescenceGPlug = PlugDescriptor("incandescenceG")
	incandescenceR_ : IncandescenceRPlug = PlugDescriptor("incandescenceR")
	ir_ : IncandescenceRPlug = PlugDescriptor("incandescenceR")
	node : VolumeFog = None
	pass
class LightAmbientPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : VolumeFog = None
	pass
class LightBlindDataPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : VolumeFog = None
	pass
class LightDiffusePlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : VolumeFog = None
	pass
class LightDirectionXPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : VolumeFog = None
	pass
class LightDirectionYPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : VolumeFog = None
	pass
class LightDirectionZPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : VolumeFog = None
	pass
class LightDirectionPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	lightDirectionX_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	ldx_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	lightDirectionY_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	ldy_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	lightDirectionZ_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	ldz_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	node : VolumeFog = None
	pass
class LightIntensityBPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : VolumeFog = None
	pass
class LightIntensityGPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : VolumeFog = None
	pass
class LightIntensityRPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : VolumeFog = None
	pass
class LightIntensityPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	lightIntensityB_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lib_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lightIntensityG_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lig_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lightIntensityR_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	lir_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	node : VolumeFog = None
	pass
class LightShadowFractionPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : VolumeFog = None
	pass
class LightSpecularPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : VolumeFog = None
	pass
class PreShadowIntensityPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : VolumeFog = None
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
	node : VolumeFog = None
	pass
class LightScatterPlug(Plug):
	node : VolumeFog = None
	pass
class MatrixWorldToEyePlug(Plug):
	node : VolumeFog = None
	pass
class MatteOpacityPlug(Plug):
	node : VolumeFog = None
	pass
class MatteOpacityModePlug(Plug):
	node : VolumeFog = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : VolumeFog = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : VolumeFog = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : VolumeFog = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : VolumeFog = None
	pass
class OutGlowColorBPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : VolumeFog = None
	pass
class OutGlowColorGPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : VolumeFog = None
	pass
class OutGlowColorRPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : VolumeFog = None
	pass
class OutGlowColorPlug(Plug):
	outGlowColorB_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	ogb_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	outGlowColorG_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	ogg_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	outGlowColorR_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	ogr_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	node : VolumeFog = None
	pass
class OutMatteOpacityBPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : VolumeFog = None
	pass
class OutMatteOpacityGPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : VolumeFog = None
	pass
class OutMatteOpacityRPlug(Plug):
	parent : OutMatteOpacityPlug = PlugDescriptor("outMatteOpacity")
	node : VolumeFog = None
	pass
class OutMatteOpacityPlug(Plug):
	outMatteOpacityB_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	omob_ : OutMatteOpacityBPlug = PlugDescriptor("outMatteOpacityB")
	outMatteOpacityG_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	omog_ : OutMatteOpacityGPlug = PlugDescriptor("outMatteOpacityG")
	outMatteOpacityR_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	omor_ : OutMatteOpacityRPlug = PlugDescriptor("outMatteOpacityR")
	node : VolumeFog = None
	pass
class OutTransparencyBPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : VolumeFog = None
	pass
class OutTransparencyGPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : VolumeFog = None
	pass
class OutTransparencyRPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : VolumeFog = None
	pass
class OutTransparencyPlug(Plug):
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	otb_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	otg_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	otr_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	node : VolumeFog = None
	pass
class PointObjXPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : VolumeFog = None
	pass
class PointObjYPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : VolumeFog = None
	pass
class PointObjZPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : VolumeFog = None
	pass
class PointObjPlug(Plug):
	pointObjX_ : PointObjXPlug = PlugDescriptor("pointObjX")
	pox_ : PointObjXPlug = PlugDescriptor("pointObjX")
	pointObjY_ : PointObjYPlug = PlugDescriptor("pointObjY")
	poy_ : PointObjYPlug = PlugDescriptor("pointObjY")
	pointObjZ_ : PointObjZPlug = PlugDescriptor("pointObjZ")
	poz_ : PointObjZPlug = PlugDescriptor("pointObjZ")
	node : VolumeFog = None
	pass
class PointWorldXPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : VolumeFog = None
	pass
class PointWorldYPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : VolumeFog = None
	pass
class PointWorldZPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : VolumeFog = None
	pass
class PointWorldPlug(Plug):
	pointWorldX_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pwx_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pointWorldY_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pwy_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pointWorldZ_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	pwz_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	node : VolumeFog = None
	pass
class RayDirectionXPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : VolumeFog = None
	pass
class RayDirectionYPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : VolumeFog = None
	pass
class RayDirectionZPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : VolumeFog = None
	pass
class RayDirectionPlug(Plug):
	rayDirectionX_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rdx_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rayDirectionY_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rdy_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rayDirectionZ_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	rdz_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	node : VolumeFog = None
	pass
class TransparencyBPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : VolumeFog = None
	pass
class TransparencyGPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : VolumeFog = None
	pass
class TransparencyRPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : VolumeFog = None
	pass
class TransparencyPlug(Plug):
	transparencyB_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	tb_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	transparencyG_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	tg_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	transparencyR_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	tr_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	node : VolumeFog = None
	pass
# endregion


# define node class
class VolumeFog(ShadingDependNode):
	axialDropoff_ : AxialDropoffPlug = PlugDescriptor("axialDropoff")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	color_ : ColorPlug = PlugDescriptor("color")
	colorRamp_ColorB_ : ColorRamp_ColorBPlug = PlugDescriptor("colorRamp_ColorB")
	colorRamp_ColorG_ : ColorRamp_ColorGPlug = PlugDescriptor("colorRamp_ColorG")
	colorRamp_ColorR_ : ColorRamp_ColorRPlug = PlugDescriptor("colorRamp_ColorR")
	colorRamp_Color_ : ColorRamp_ColorPlug = PlugDescriptor("colorRamp_Color")
	colorRamp_Interp_ : ColorRamp_InterpPlug = PlugDescriptor("colorRamp_Interp")
	colorRamp_Position_ : ColorRamp_PositionPlug = PlugDescriptor("colorRamp_Position")
	colorRamp_ : ColorRampPlug = PlugDescriptor("colorRamp")
	colorRampInput_ : ColorRampInputPlug = PlugDescriptor("colorRampInput")
	density_ : DensityPlug = PlugDescriptor("density")
	densityMode_ : DensityModePlug = PlugDescriptor("densityMode")
	dropoffMethod_ : DropoffMethodPlug = PlugDescriptor("dropoffMethod")
	dropoffShape_ : DropoffShapePlug = PlugDescriptor("dropoffShape")
	dropoffSubtract_ : DropoffSubtractPlug = PlugDescriptor("dropoffSubtract")
	edgeDropoff_ : EdgeDropoffPlug = PlugDescriptor("edgeDropoff")
	farPointObjectX_ : FarPointObjectXPlug = PlugDescriptor("farPointObjectX")
	farPointObjectY_ : FarPointObjectYPlug = PlugDescriptor("farPointObjectY")
	farPointObjectZ_ : FarPointObjectZPlug = PlugDescriptor("farPointObjectZ")
	farPointObj_ : FarPointObjPlug = PlugDescriptor("farPointObj")
	farPointWorldX_ : FarPointWorldXPlug = PlugDescriptor("farPointWorldX")
	farPointWorldY_ : FarPointWorldYPlug = PlugDescriptor("farPointWorldY")
	farPointWorldZ_ : FarPointWorldZPlug = PlugDescriptor("farPointWorldZ")
	farPointWorld_ : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	glowIntensity_ : GlowIntensityPlug = PlugDescriptor("glowIntensity")
	illuminated_ : IlluminatedPlug = PlugDescriptor("illuminated")
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
	lightScatter_ : LightScatterPlug = PlugDescriptor("lightScatter")
	matrixWorldToEye_ : MatrixWorldToEyePlug = PlugDescriptor("matrixWorldToEye")
	matteOpacity_ : MatteOpacityPlug = PlugDescriptor("matteOpacity")
	matteOpacityMode_ : MatteOpacityModePlug = PlugDescriptor("matteOpacityMode")
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
	transparencyB_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	transparencyG_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	transparencyR_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	transparency_ : TransparencyPlug = PlugDescriptor("transparency")

	# node attributes

	typeName = "volumeFog"
	apiTypeInt = 870
	apiTypeStr = "kVolumeFog"
	typeIdInt = 1381385799
	MFnCls = om.MFnDependencyNode
	pass

