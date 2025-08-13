

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
class BlobMapBPlug(Plug):
	parent : BlobMapPlug = PlugDescriptor("blobMap")
	node : ParticleCloud = None
	pass
class BlobMapGPlug(Plug):
	parent : BlobMapPlug = PlugDescriptor("blobMap")
	node : ParticleCloud = None
	pass
class BlobMapRPlug(Plug):
	parent : BlobMapPlug = PlugDescriptor("blobMap")
	node : ParticleCloud = None
	pass
class BlobMapPlug(Plug):
	blobMapB_ : BlobMapBPlug = PlugDescriptor("blobMapB")
	mb_ : BlobMapBPlug = PlugDescriptor("blobMapB")
	blobMapG_ : BlobMapGPlug = PlugDescriptor("blobMapG")
	mg_ : BlobMapGPlug = PlugDescriptor("blobMapG")
	blobMapR_ : BlobMapRPlug = PlugDescriptor("blobMapR")
	mr_ : BlobMapRPlug = PlugDescriptor("blobMapR")
	node : ParticleCloud = None
	pass
class ColorBPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : ParticleCloud = None
	pass
class ColorGPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : ParticleCloud = None
	pass
class ColorRPlug(Plug):
	parent : ColorPlug = PlugDescriptor("color")
	node : ParticleCloud = None
	pass
class ColorPlug(Plug):
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	cb_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	cg_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	cr_ : ColorRPlug = PlugDescriptor("colorR")
	node : ParticleCloud = None
	pass
class DensityPlug(Plug):
	node : ParticleCloud = None
	pass
class DiffuseCoeffPlug(Plug):
	node : ParticleCloud = None
	pass
class FilterRadiusPlug(Plug):
	node : ParticleCloud = None
	pass
class GlowIntensityPlug(Plug):
	node : ParticleCloud = None
	pass
class IncandescenceBPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : ParticleCloud = None
	pass
class IncandescenceGPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : ParticleCloud = None
	pass
class IncandescenceRPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : ParticleCloud = None
	pass
class IncandescencePlug(Plug):
	incandescenceB_ : IncandescenceBPlug = PlugDescriptor("incandescenceB")
	ib_ : IncandescenceBPlug = PlugDescriptor("incandescenceB")
	incandescenceG_ : IncandescenceGPlug = PlugDescriptor("incandescenceG")
	ig_ : IncandescenceGPlug = PlugDescriptor("incandescenceG")
	incandescenceR_ : IncandescenceRPlug = PlugDescriptor("incandescenceR")
	ir_ : IncandescenceRPlug = PlugDescriptor("incandescenceR")
	node : ParticleCloud = None
	pass
class LightAmbientPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : ParticleCloud = None
	pass
class LightBlindDataPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : ParticleCloud = None
	pass
class LightDiffusePlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : ParticleCloud = None
	pass
class LightDirectionXPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : ParticleCloud = None
	pass
class LightDirectionYPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : ParticleCloud = None
	pass
class LightDirectionZPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : ParticleCloud = None
	pass
class LightDirectionPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	lightDirectionX_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	ldx_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	lightDirectionY_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	ldy_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	lightDirectionZ_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	ldz_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	node : ParticleCloud = None
	pass
class LightIntensityBPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : ParticleCloud = None
	pass
class LightIntensityGPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : ParticleCloud = None
	pass
class LightIntensityRPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : ParticleCloud = None
	pass
class LightIntensityPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	lightIntensityB_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lib_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lightIntensityG_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lig_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lightIntensityR_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	lir_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	node : ParticleCloud = None
	pass
class LightShadowFractionPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : ParticleCloud = None
	pass
class LightSpecularPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : ParticleCloud = None
	pass
class PreShadowIntensityPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : ParticleCloud = None
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
	node : ParticleCloud = None
	pass
class NoisePlug(Plug):
	node : ParticleCloud = None
	pass
class NoiseAnimRatePlug(Plug):
	node : ParticleCloud = None
	pass
class NoiseAspectPlug(Plug):
	node : ParticleCloud = None
	pass
class NoiseFreqPlug(Plug):
	node : ParticleCloud = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : ParticleCloud = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : ParticleCloud = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : ParticleCloud = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	ncx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ncy_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	ncz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : ParticleCloud = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : ParticleCloud = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : ParticleCloud = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : ParticleCloud = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	oib_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	oig_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	oir_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : ParticleCloud = None
	pass
class OutGlowColorBPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : ParticleCloud = None
	pass
class OutGlowColorGPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : ParticleCloud = None
	pass
class OutGlowColorRPlug(Plug):
	parent : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	node : ParticleCloud = None
	pass
class OutGlowColorPlug(Plug):
	outGlowColorB_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	ogb_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	outGlowColorG_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	ogg_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	outGlowColorR_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	ogr_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	node : ParticleCloud = None
	pass
class OutParticleEmissionBPlug(Plug):
	parent : OutParticleEmissionPlug = PlugDescriptor("outParticleEmission")
	node : ParticleCloud = None
	pass
class OutParticleEmissionGPlug(Plug):
	parent : OutParticleEmissionPlug = PlugDescriptor("outParticleEmission")
	node : ParticleCloud = None
	pass
class OutParticleEmissionRPlug(Plug):
	parent : OutParticleEmissionPlug = PlugDescriptor("outParticleEmission")
	node : ParticleCloud = None
	pass
class OutParticleEmissionPlug(Plug):
	outParticleEmissionB_ : OutParticleEmissionBPlug = PlugDescriptor("outParticleEmissionB")
	oeb_ : OutParticleEmissionBPlug = PlugDescriptor("outParticleEmissionB")
	outParticleEmissionG_ : OutParticleEmissionGPlug = PlugDescriptor("outParticleEmissionG")
	oeg_ : OutParticleEmissionGPlug = PlugDescriptor("outParticleEmissionG")
	outParticleEmissionR_ : OutParticleEmissionRPlug = PlugDescriptor("outParticleEmissionR")
	oer_ : OutParticleEmissionRPlug = PlugDescriptor("outParticleEmissionR")
	node : ParticleCloud = None
	pass
class OutTransparencyBPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : ParticleCloud = None
	pass
class OutTransparencyGPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : ParticleCloud = None
	pass
class OutTransparencyRPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : ParticleCloud = None
	pass
class OutTransparencyPlug(Plug):
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	otb_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	otg_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	otr_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	node : ParticleCloud = None
	pass
class ParticleEmissionBPlug(Plug):
	parent : ParticleEmissionPlug = PlugDescriptor("particleEmission")
	node : ParticleCloud = None
	pass
class ParticleEmissionGPlug(Plug):
	parent : ParticleEmissionPlug = PlugDescriptor("particleEmission")
	node : ParticleCloud = None
	pass
class ParticleEmissionRPlug(Plug):
	parent : ParticleEmissionPlug = PlugDescriptor("particleEmission")
	node : ParticleCloud = None
	pass
class ParticleEmissionPlug(Plug):
	particleEmissionB_ : ParticleEmissionBPlug = PlugDescriptor("particleEmissionB")
	eb_ : ParticleEmissionBPlug = PlugDescriptor("particleEmissionB")
	particleEmissionG_ : ParticleEmissionGPlug = PlugDescriptor("particleEmissionG")
	eg_ : ParticleEmissionGPlug = PlugDescriptor("particleEmissionG")
	particleEmissionR_ : ParticleEmissionRPlug = PlugDescriptor("particleEmissionR")
	er_ : ParticleEmissionRPlug = PlugDescriptor("particleEmissionR")
	node : ParticleCloud = None
	pass
class ParticleOrderPlug(Plug):
	node : ParticleCloud = None
	pass
class ParticleWeightPlug(Plug):
	node : ParticleCloud = None
	pass
class PointObjXPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : ParticleCloud = None
	pass
class PointObjYPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : ParticleCloud = None
	pass
class PointObjZPlug(Plug):
	parent : PointObjPlug = PlugDescriptor("pointObj")
	node : ParticleCloud = None
	pass
class PointObjPlug(Plug):
	pointObjX_ : PointObjXPlug = PlugDescriptor("pointObjX")
	px_ : PointObjXPlug = PlugDescriptor("pointObjX")
	pointObjY_ : PointObjYPlug = PlugDescriptor("pointObjY")
	py_ : PointObjYPlug = PlugDescriptor("pointObjY")
	pointObjZ_ : PointObjZPlug = PlugDescriptor("pointObjZ")
	pz_ : PointObjZPlug = PlugDescriptor("pointObjZ")
	node : ParticleCloud = None
	pass
class RayDepthPlug(Plug):
	node : ParticleCloud = None
	pass
class RenderStatePlug(Plug):
	node : ParticleCloud = None
	pass
class RoundnessPlug(Plug):
	node : ParticleCloud = None
	pass
class SolidCoreSizePlug(Plug):
	node : ParticleCloud = None
	pass
class SurfaceColorBPlug(Plug):
	parent : SurfaceColorPlug = PlugDescriptor("surfaceColor")
	node : ParticleCloud = None
	pass
class SurfaceColorGPlug(Plug):
	parent : SurfaceColorPlug = PlugDescriptor("surfaceColor")
	node : ParticleCloud = None
	pass
class SurfaceColorRPlug(Plug):
	parent : SurfaceColorPlug = PlugDescriptor("surfaceColor")
	node : ParticleCloud = None
	pass
class SurfaceColorPlug(Plug):
	surfaceColorB_ : SurfaceColorBPlug = PlugDescriptor("surfaceColorB")
	scb_ : SurfaceColorBPlug = PlugDescriptor("surfaceColorB")
	surfaceColorG_ : SurfaceColorGPlug = PlugDescriptor("surfaceColorG")
	scg_ : SurfaceColorGPlug = PlugDescriptor("surfaceColorG")
	surfaceColorR_ : SurfaceColorRPlug = PlugDescriptor("surfaceColorR")
	scr_ : SurfaceColorRPlug = PlugDescriptor("surfaceColorR")
	node : ParticleCloud = None
	pass
class SurfaceShadingShadowPlug(Plug):
	node : ParticleCloud = None
	pass
class TranslucencePlug(Plug):
	node : ParticleCloud = None
	pass
class TranslucenceCoeffPlug(Plug):
	node : ParticleCloud = None
	pass
class TransparencyBPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : ParticleCloud = None
	pass
class TransparencyGPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : ParticleCloud = None
	pass
class TransparencyRPlug(Plug):
	parent : TransparencyPlug = PlugDescriptor("transparency")
	node : ParticleCloud = None
	pass
class TransparencyPlug(Plug):
	transparencyB_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	tb_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	transparencyG_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	tg_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	transparencyR_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	tr_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	node : ParticleCloud = None
	pass
# endregion


# define node class
class ParticleCloud(ShadingDependNode):
	blobMapB_ : BlobMapBPlug = PlugDescriptor("blobMapB")
	blobMapG_ : BlobMapGPlug = PlugDescriptor("blobMapG")
	blobMapR_ : BlobMapRPlug = PlugDescriptor("blobMapR")
	blobMap_ : BlobMapPlug = PlugDescriptor("blobMap")
	colorB_ : ColorBPlug = PlugDescriptor("colorB")
	colorG_ : ColorGPlug = PlugDescriptor("colorG")
	colorR_ : ColorRPlug = PlugDescriptor("colorR")
	color_ : ColorPlug = PlugDescriptor("color")
	density_ : DensityPlug = PlugDescriptor("density")
	diffuseCoeff_ : DiffuseCoeffPlug = PlugDescriptor("diffuseCoeff")
	filterRadius_ : FilterRadiusPlug = PlugDescriptor("filterRadius")
	glowIntensity_ : GlowIntensityPlug = PlugDescriptor("glowIntensity")
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
	noise_ : NoisePlug = PlugDescriptor("noise")
	noiseAnimRate_ : NoiseAnimRatePlug = PlugDescriptor("noiseAnimRate")
	noiseAspect_ : NoiseAspectPlug = PlugDescriptor("noiseAspect")
	noiseFreq_ : NoiseFreqPlug = PlugDescriptor("noiseFreq")
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	normalCamera_ : NormalCameraPlug = PlugDescriptor("normalCamera")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	outGlowColorB_ : OutGlowColorBPlug = PlugDescriptor("outGlowColorB")
	outGlowColorG_ : OutGlowColorGPlug = PlugDescriptor("outGlowColorG")
	outGlowColorR_ : OutGlowColorRPlug = PlugDescriptor("outGlowColorR")
	outGlowColor_ : OutGlowColorPlug = PlugDescriptor("outGlowColor")
	outParticleEmissionB_ : OutParticleEmissionBPlug = PlugDescriptor("outParticleEmissionB")
	outParticleEmissionG_ : OutParticleEmissionGPlug = PlugDescriptor("outParticleEmissionG")
	outParticleEmissionR_ : OutParticleEmissionRPlug = PlugDescriptor("outParticleEmissionR")
	outParticleEmission_ : OutParticleEmissionPlug = PlugDescriptor("outParticleEmission")
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	outTransparency_ : OutTransparencyPlug = PlugDescriptor("outTransparency")
	particleEmissionB_ : ParticleEmissionBPlug = PlugDescriptor("particleEmissionB")
	particleEmissionG_ : ParticleEmissionGPlug = PlugDescriptor("particleEmissionG")
	particleEmissionR_ : ParticleEmissionRPlug = PlugDescriptor("particleEmissionR")
	particleEmission_ : ParticleEmissionPlug = PlugDescriptor("particleEmission")
	particleOrder_ : ParticleOrderPlug = PlugDescriptor("particleOrder")
	particleWeight_ : ParticleWeightPlug = PlugDescriptor("particleWeight")
	pointObjX_ : PointObjXPlug = PlugDescriptor("pointObjX")
	pointObjY_ : PointObjYPlug = PlugDescriptor("pointObjY")
	pointObjZ_ : PointObjZPlug = PlugDescriptor("pointObjZ")
	pointObj_ : PointObjPlug = PlugDescriptor("pointObj")
	rayDepth_ : RayDepthPlug = PlugDescriptor("rayDepth")
	renderState_ : RenderStatePlug = PlugDescriptor("renderState")
	roundness_ : RoundnessPlug = PlugDescriptor("roundness")
	solidCoreSize_ : SolidCoreSizePlug = PlugDescriptor("solidCoreSize")
	surfaceColorB_ : SurfaceColorBPlug = PlugDescriptor("surfaceColorB")
	surfaceColorG_ : SurfaceColorGPlug = PlugDescriptor("surfaceColorG")
	surfaceColorR_ : SurfaceColorRPlug = PlugDescriptor("surfaceColorR")
	surfaceColor_ : SurfaceColorPlug = PlugDescriptor("surfaceColor")
	surfaceShadingShadow_ : SurfaceShadingShadowPlug = PlugDescriptor("surfaceShadingShadow")
	translucence_ : TranslucencePlug = PlugDescriptor("translucence")
	translucenceCoeff_ : TranslucenceCoeffPlug = PlugDescriptor("translucenceCoeff")
	transparencyB_ : TransparencyBPlug = PlugDescriptor("transparencyB")
	transparencyG_ : TransparencyGPlug = PlugDescriptor("transparencyG")
	transparencyR_ : TransparencyRPlug = PlugDescriptor("transparencyR")
	transparency_ : TransparencyPlug = PlugDescriptor("transparency")

	# node attributes

	typeName = "particleCloud"
	apiTypeInt = 452
	apiTypeStr = "kParticleCloud"
	typeIdInt = 1346587716
	MFnCls = om.MFnDependencyNode
	pass

