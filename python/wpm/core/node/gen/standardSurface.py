

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
class BasePlug(Plug):
	node : StandardSurface = None
	pass
class BaseColorBPlug(Plug):
	parent : BaseColorPlug = PlugDescriptor("baseColor")
	node : StandardSurface = None
	pass
class BaseColorGPlug(Plug):
	parent : BaseColorPlug = PlugDescriptor("baseColor")
	node : StandardSurface = None
	pass
class BaseColorRPlug(Plug):
	parent : BaseColorPlug = PlugDescriptor("baseColor")
	node : StandardSurface = None
	pass
class BaseColorPlug(Plug):
	baseColorB_ : BaseColorBPlug = PlugDescriptor("baseColorB")
	bcb_ : BaseColorBPlug = PlugDescriptor("baseColorB")
	baseColorG_ : BaseColorGPlug = PlugDescriptor("baseColorG")
	bcg_ : BaseColorGPlug = PlugDescriptor("baseColorG")
	baseColorR_ : BaseColorRPlug = PlugDescriptor("baseColorR")
	bcr_ : BaseColorRPlug = PlugDescriptor("baseColorR")
	node : StandardSurface = None
	pass
class CoatPlug(Plug):
	node : StandardSurface = None
	pass
class CoatAffectColorPlug(Plug):
	node : StandardSurface = None
	pass
class CoatAffectRoughnessPlug(Plug):
	node : StandardSurface = None
	pass
class CoatAnisotropyPlug(Plug):
	node : StandardSurface = None
	pass
class CoatColorBPlug(Plug):
	parent : CoatColorPlug = PlugDescriptor("coatColor")
	node : StandardSurface = None
	pass
class CoatColorGPlug(Plug):
	parent : CoatColorPlug = PlugDescriptor("coatColor")
	node : StandardSurface = None
	pass
class CoatColorRPlug(Plug):
	parent : CoatColorPlug = PlugDescriptor("coatColor")
	node : StandardSurface = None
	pass
class CoatColorPlug(Plug):
	coatColorB_ : CoatColorBPlug = PlugDescriptor("coatColorB")
	ctcb_ : CoatColorBPlug = PlugDescriptor("coatColorB")
	coatColorG_ : CoatColorGPlug = PlugDescriptor("coatColorG")
	ctcg_ : CoatColorGPlug = PlugDescriptor("coatColorG")
	coatColorR_ : CoatColorRPlug = PlugDescriptor("coatColorR")
	ctcr_ : CoatColorRPlug = PlugDescriptor("coatColorR")
	node : StandardSurface = None
	pass
class CoatIORPlug(Plug):
	node : StandardSurface = None
	pass
class CoatNormalXPlug(Plug):
	parent : CoatNormalPlug = PlugDescriptor("coatNormal")
	node : StandardSurface = None
	pass
class CoatNormalYPlug(Plug):
	parent : CoatNormalPlug = PlugDescriptor("coatNormal")
	node : StandardSurface = None
	pass
class CoatNormalZPlug(Plug):
	parent : CoatNormalPlug = PlugDescriptor("coatNormal")
	node : StandardSurface = None
	pass
class CoatNormalPlug(Plug):
	coatNormalX_ : CoatNormalXPlug = PlugDescriptor("coatNormalX")
	ctnx_ : CoatNormalXPlug = PlugDescriptor("coatNormalX")
	coatNormalY_ : CoatNormalYPlug = PlugDescriptor("coatNormalY")
	ctny_ : CoatNormalYPlug = PlugDescriptor("coatNormalY")
	coatNormalZ_ : CoatNormalZPlug = PlugDescriptor("coatNormalZ")
	ctnz_ : CoatNormalZPlug = PlugDescriptor("coatNormalZ")
	node : StandardSurface = None
	pass
class CoatRotationPlug(Plug):
	node : StandardSurface = None
	pass
class CoatRoughnessPlug(Plug):
	node : StandardSurface = None
	pass
class DiffuseRoughnessPlug(Plug):
	node : StandardSurface = None
	pass
class EmissionPlug(Plug):
	node : StandardSurface = None
	pass
class EmissionColorBPlug(Plug):
	parent : EmissionColorPlug = PlugDescriptor("emissionColor")
	node : StandardSurface = None
	pass
class EmissionColorGPlug(Plug):
	parent : EmissionColorPlug = PlugDescriptor("emissionColor")
	node : StandardSurface = None
	pass
class EmissionColorRPlug(Plug):
	parent : EmissionColorPlug = PlugDescriptor("emissionColor")
	node : StandardSurface = None
	pass
class EmissionColorPlug(Plug):
	emissionColorB_ : EmissionColorBPlug = PlugDescriptor("emissionColorB")
	ecb_ : EmissionColorBPlug = PlugDescriptor("emissionColorB")
	emissionColorG_ : EmissionColorGPlug = PlugDescriptor("emissionColorG")
	ecg_ : EmissionColorGPlug = PlugDescriptor("emissionColorG")
	emissionColorR_ : EmissionColorRPlug = PlugDescriptor("emissionColorR")
	ecr_ : EmissionColorRPlug = PlugDescriptor("emissionColorR")
	node : StandardSurface = None
	pass
class LightAmbientPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : StandardSurface = None
	pass
class LightBlindDataPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : StandardSurface = None
	pass
class LightDiffusePlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : StandardSurface = None
	pass
class LightDirectionXPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : StandardSurface = None
	pass
class LightDirectionYPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : StandardSurface = None
	pass
class LightDirectionZPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : StandardSurface = None
	pass
class LightDirectionPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	lightDirectionX_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	ldx_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	lightDirectionY_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	ldy_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	lightDirectionZ_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	ldz_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	node : StandardSurface = None
	pass
class LightIntensityBPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : StandardSurface = None
	pass
class LightIntensityGPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : StandardSurface = None
	pass
class LightIntensityRPlug(Plug):
	parent : LightIntensityPlug = PlugDescriptor("lightIntensity")
	node : StandardSurface = None
	pass
class LightIntensityPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	lightIntensityB_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lib_ : LightIntensityBPlug = PlugDescriptor("lightIntensityB")
	lightIntensityG_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lig_ : LightIntensityGPlug = PlugDescriptor("lightIntensityG")
	lightIntensityR_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	lir_ : LightIntensityRPlug = PlugDescriptor("lightIntensityR")
	node : StandardSurface = None
	pass
class LightShadowFractionPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : StandardSurface = None
	pass
class LightSpecularPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : StandardSurface = None
	pass
class PreShadowIntensityPlug(Plug):
	parent : LightDataArrayPlug = PlugDescriptor("lightDataArray")
	node : StandardSurface = None
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
	node : StandardSurface = None
	pass
class MetalnessPlug(Plug):
	node : StandardSurface = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : StandardSurface = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : StandardSurface = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : StandardSurface = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	nx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ny_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	nz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : StandardSurface = None
	pass
class ObjectIdPlug(Plug):
	node : StandardSurface = None
	pass
class OpacityBPlug(Plug):
	parent : OpacityPlug = PlugDescriptor("opacity")
	node : StandardSurface = None
	pass
class OpacityGPlug(Plug):
	parent : OpacityPlug = PlugDescriptor("opacity")
	node : StandardSurface = None
	pass
class OpacityRPlug(Plug):
	parent : OpacityPlug = PlugDescriptor("opacity")
	node : StandardSurface = None
	pass
class OpacityPlug(Plug):
	opacityB_ : OpacityBPlug = PlugDescriptor("opacityB")
	opb_ : OpacityBPlug = PlugDescriptor("opacityB")
	opacityG_ : OpacityGPlug = PlugDescriptor("opacityG")
	opg_ : OpacityGPlug = PlugDescriptor("opacityG")
	opacityR_ : OpacityRPlug = PlugDescriptor("opacityR")
	opr_ : OpacityRPlug = PlugDescriptor("opacityR")
	node : StandardSurface = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : StandardSurface = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : StandardSurface = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : StandardSurface = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : StandardSurface = None
	pass
class OutTransparencyBPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : StandardSurface = None
	pass
class OutTransparencyGPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : StandardSurface = None
	pass
class OutTransparencyRPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : StandardSurface = None
	pass
class OutTransparencyPlug(Plug):
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	otb_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	otg_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	otr_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	node : StandardSurface = None
	pass
class PointCameraXPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : StandardSurface = None
	pass
class PointCameraYPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : StandardSurface = None
	pass
class PointCameraZPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : StandardSurface = None
	pass
class PointCameraPlug(Plug):
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	px_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	py_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pz_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	node : StandardSurface = None
	pass
class PrimitiveIdPlug(Plug):
	node : StandardSurface = None
	pass
class RayDepthPlug(Plug):
	node : StandardSurface = None
	pass
class RayDirectionXPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : StandardSurface = None
	pass
class RayDirectionYPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : StandardSurface = None
	pass
class RayDirectionZPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : StandardSurface = None
	pass
class RayDirectionPlug(Plug):
	rayDirectionX_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rdx_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rayDirectionY_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rdy_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rayDirectionZ_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	rdz_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	node : StandardSurface = None
	pass
class RayInstancePlug(Plug):
	node : StandardSurface = None
	pass
class RaySamplerPlug(Plug):
	node : StandardSurface = None
	pass
class SheenPlug(Plug):
	node : StandardSurface = None
	pass
class SheenColorBPlug(Plug):
	parent : SheenColorPlug = PlugDescriptor("sheenColor")
	node : StandardSurface = None
	pass
class SheenColorGPlug(Plug):
	parent : SheenColorPlug = PlugDescriptor("sheenColor")
	node : StandardSurface = None
	pass
class SheenColorRPlug(Plug):
	parent : SheenColorPlug = PlugDescriptor("sheenColor")
	node : StandardSurface = None
	pass
class SheenColorPlug(Plug):
	sheenColorB_ : SheenColorBPlug = PlugDescriptor("sheenColorB")
	shcb_ : SheenColorBPlug = PlugDescriptor("sheenColorB")
	sheenColorG_ : SheenColorGPlug = PlugDescriptor("sheenColorG")
	shcg_ : SheenColorGPlug = PlugDescriptor("sheenColorG")
	sheenColorR_ : SheenColorRPlug = PlugDescriptor("sheenColorR")
	shcr_ : SheenColorRPlug = PlugDescriptor("sheenColorR")
	node : StandardSurface = None
	pass
class SheenRoughnessPlug(Plug):
	node : StandardSurface = None
	pass
class SpecularPlug(Plug):
	node : StandardSurface = None
	pass
class SpecularAnisotropyPlug(Plug):
	node : StandardSurface = None
	pass
class SpecularColorBPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : StandardSurface = None
	pass
class SpecularColorGPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : StandardSurface = None
	pass
class SpecularColorRPlug(Plug):
	parent : SpecularColorPlug = PlugDescriptor("specularColor")
	node : StandardSurface = None
	pass
class SpecularColorPlug(Plug):
	specularColorB_ : SpecularColorBPlug = PlugDescriptor("specularColorB")
	spb_ : SpecularColorBPlug = PlugDescriptor("specularColorB")
	specularColorG_ : SpecularColorGPlug = PlugDescriptor("specularColorG")
	scg_ : SpecularColorGPlug = PlugDescriptor("specularColorG")
	specularColorR_ : SpecularColorRPlug = PlugDescriptor("specularColorR")
	scr_ : SpecularColorRPlug = PlugDescriptor("specularColorR")
	node : StandardSurface = None
	pass
class SpecularIORPlug(Plug):
	node : StandardSurface = None
	pass
class SpecularRotationPlug(Plug):
	node : StandardSurface = None
	pass
class SpecularRoughnessPlug(Plug):
	node : StandardSurface = None
	pass
class SubsurfacePlug(Plug):
	node : StandardSurface = None
	pass
class SubsurfaceAnisotropyPlug(Plug):
	node : StandardSurface = None
	pass
class SubsurfaceColorBPlug(Plug):
	parent : SubsurfaceColorPlug = PlugDescriptor("subsurfaceColor")
	node : StandardSurface = None
	pass
class SubsurfaceColorGPlug(Plug):
	parent : SubsurfaceColorPlug = PlugDescriptor("subsurfaceColor")
	node : StandardSurface = None
	pass
class SubsurfaceColorRPlug(Plug):
	parent : SubsurfaceColorPlug = PlugDescriptor("subsurfaceColor")
	node : StandardSurface = None
	pass
class SubsurfaceColorPlug(Plug):
	subsurfaceColorB_ : SubsurfaceColorBPlug = PlugDescriptor("subsurfaceColorB")
	subcb_ : SubsurfaceColorBPlug = PlugDescriptor("subsurfaceColorB")
	subsurfaceColorG_ : SubsurfaceColorGPlug = PlugDescriptor("subsurfaceColorG")
	subcg_ : SubsurfaceColorGPlug = PlugDescriptor("subsurfaceColorG")
	subsurfaceColorR_ : SubsurfaceColorRPlug = PlugDescriptor("subsurfaceColorR")
	subcr_ : SubsurfaceColorRPlug = PlugDescriptor("subsurfaceColorR")
	node : StandardSurface = None
	pass
class SubsurfaceRadiusBPlug(Plug):
	parent : SubsurfaceRadiusPlug = PlugDescriptor("subsurfaceRadius")
	node : StandardSurface = None
	pass
class SubsurfaceRadiusGPlug(Plug):
	parent : SubsurfaceRadiusPlug = PlugDescriptor("subsurfaceRadius")
	node : StandardSurface = None
	pass
class SubsurfaceRadiusRPlug(Plug):
	parent : SubsurfaceRadiusPlug = PlugDescriptor("subsurfaceRadius")
	node : StandardSurface = None
	pass
class SubsurfaceRadiusPlug(Plug):
	subsurfaceRadiusB_ : SubsurfaceRadiusBPlug = PlugDescriptor("subsurfaceRadiusB")
	subrb_ : SubsurfaceRadiusBPlug = PlugDescriptor("subsurfaceRadiusB")
	subsurfaceRadiusG_ : SubsurfaceRadiusGPlug = PlugDescriptor("subsurfaceRadiusG")
	subrg_ : SubsurfaceRadiusGPlug = PlugDescriptor("subsurfaceRadiusG")
	subsurfaceRadiusR_ : SubsurfaceRadiusRPlug = PlugDescriptor("subsurfaceRadiusR")
	subrr_ : SubsurfaceRadiusRPlug = PlugDescriptor("subsurfaceRadiusR")
	node : StandardSurface = None
	pass
class SubsurfaceScalePlug(Plug):
	node : StandardSurface = None
	pass
class TangentUCameraXPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : StandardSurface = None
	pass
class TangentUCameraYPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : StandardSurface = None
	pass
class TangentUCameraZPlug(Plug):
	parent : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	node : StandardSurface = None
	pass
class TangentUCameraPlug(Plug):
	tangentUCameraX_ : TangentUCameraXPlug = PlugDescriptor("tangentUCameraX")
	utnx_ : TangentUCameraXPlug = PlugDescriptor("tangentUCameraX")
	tangentUCameraY_ : TangentUCameraYPlug = PlugDescriptor("tangentUCameraY")
	utny_ : TangentUCameraYPlug = PlugDescriptor("tangentUCameraY")
	tangentUCameraZ_ : TangentUCameraZPlug = PlugDescriptor("tangentUCameraZ")
	utnz_ : TangentUCameraZPlug = PlugDescriptor("tangentUCameraZ")
	node : StandardSurface = None
	pass
class ThinFilmIORPlug(Plug):
	node : StandardSurface = None
	pass
class ThinFilmThicknessPlug(Plug):
	node : StandardSurface = None
	pass
class ThinWalledPlug(Plug):
	node : StandardSurface = None
	pass
class TransmissionPlug(Plug):
	node : StandardSurface = None
	pass
class TransmissionColorBPlug(Plug):
	parent : TransmissionColorPlug = PlugDescriptor("transmissionColor")
	node : StandardSurface = None
	pass
class TransmissionColorGPlug(Plug):
	parent : TransmissionColorPlug = PlugDescriptor("transmissionColor")
	node : StandardSurface = None
	pass
class TransmissionColorRPlug(Plug):
	parent : TransmissionColorPlug = PlugDescriptor("transmissionColor")
	node : StandardSurface = None
	pass
class TransmissionColorPlug(Plug):
	transmissionColorB_ : TransmissionColorBPlug = PlugDescriptor("transmissionColorB")
	trcb_ : TransmissionColorBPlug = PlugDescriptor("transmissionColorB")
	transmissionColorG_ : TransmissionColorGPlug = PlugDescriptor("transmissionColorG")
	trcg_ : TransmissionColorGPlug = PlugDescriptor("transmissionColorG")
	transmissionColorR_ : TransmissionColorRPlug = PlugDescriptor("transmissionColorR")
	trcr_ : TransmissionColorRPlug = PlugDescriptor("transmissionColorR")
	node : StandardSurface = None
	pass
class TransmissionDepthPlug(Plug):
	node : StandardSurface = None
	pass
class TransmissionDispersionPlug(Plug):
	node : StandardSurface = None
	pass
class TransmissionExtraRoughnessPlug(Plug):
	node : StandardSurface = None
	pass
class TransmissionScatterBPlug(Plug):
	parent : TransmissionScatterPlug = PlugDescriptor("transmissionScatter")
	node : StandardSurface = None
	pass
class TransmissionScatterGPlug(Plug):
	parent : TransmissionScatterPlug = PlugDescriptor("transmissionScatter")
	node : StandardSurface = None
	pass
class TransmissionScatterRPlug(Plug):
	parent : TransmissionScatterPlug = PlugDescriptor("transmissionScatter")
	node : StandardSurface = None
	pass
class TransmissionScatterPlug(Plug):
	transmissionScatterB_ : TransmissionScatterBPlug = PlugDescriptor("transmissionScatterB")
	tsb_ : TransmissionScatterBPlug = PlugDescriptor("transmissionScatterB")
	transmissionScatterG_ : TransmissionScatterGPlug = PlugDescriptor("transmissionScatterG")
	tsg_ : TransmissionScatterGPlug = PlugDescriptor("transmissionScatterG")
	transmissionScatterR_ : TransmissionScatterRPlug = PlugDescriptor("transmissionScatterR")
	tsr_ : TransmissionScatterRPlug = PlugDescriptor("transmissionScatterR")
	node : StandardSurface = None
	pass
class TransmissionScatterAnisotropyPlug(Plug):
	node : StandardSurface = None
	pass
class TriangleNormalCameraXPlug(Plug):
	parent : TriangleNormalCameraPlug = PlugDescriptor("triangleNormalCamera")
	node : StandardSurface = None
	pass
class TriangleNormalCameraYPlug(Plug):
	parent : TriangleNormalCameraPlug = PlugDescriptor("triangleNormalCamera")
	node : StandardSurface = None
	pass
class TriangleNormalCameraZPlug(Plug):
	parent : TriangleNormalCameraPlug = PlugDescriptor("triangleNormalCamera")
	node : StandardSurface = None
	pass
class TriangleNormalCameraPlug(Plug):
	triangleNormalCameraX_ : TriangleNormalCameraXPlug = PlugDescriptor("triangleNormalCameraX")
	tnx_ : TriangleNormalCameraXPlug = PlugDescriptor("triangleNormalCameraX")
	triangleNormalCameraY_ : TriangleNormalCameraYPlug = PlugDescriptor("triangleNormalCameraY")
	tny_ : TriangleNormalCameraYPlug = PlugDescriptor("triangleNormalCameraY")
	triangleNormalCameraZ_ : TriangleNormalCameraZPlug = PlugDescriptor("triangleNormalCameraZ")
	tnz_ : TriangleNormalCameraZPlug = PlugDescriptor("triangleNormalCameraZ")
	node : StandardSurface = None
	pass
# endregion


# define node class
class StandardSurface(PaintableShadingDependNode):
	base_ : BasePlug = PlugDescriptor("base")
	baseColorB_ : BaseColorBPlug = PlugDescriptor("baseColorB")
	baseColorG_ : BaseColorGPlug = PlugDescriptor("baseColorG")
	baseColorR_ : BaseColorRPlug = PlugDescriptor("baseColorR")
	baseColor_ : BaseColorPlug = PlugDescriptor("baseColor")
	coat_ : CoatPlug = PlugDescriptor("coat")
	coatAffectColor_ : CoatAffectColorPlug = PlugDescriptor("coatAffectColor")
	coatAffectRoughness_ : CoatAffectRoughnessPlug = PlugDescriptor("coatAffectRoughness")
	coatAnisotropy_ : CoatAnisotropyPlug = PlugDescriptor("coatAnisotropy")
	coatColorB_ : CoatColorBPlug = PlugDescriptor("coatColorB")
	coatColorG_ : CoatColorGPlug = PlugDescriptor("coatColorG")
	coatColorR_ : CoatColorRPlug = PlugDescriptor("coatColorR")
	coatColor_ : CoatColorPlug = PlugDescriptor("coatColor")
	coatIOR_ : CoatIORPlug = PlugDescriptor("coatIOR")
	coatNormalX_ : CoatNormalXPlug = PlugDescriptor("coatNormalX")
	coatNormalY_ : CoatNormalYPlug = PlugDescriptor("coatNormalY")
	coatNormalZ_ : CoatNormalZPlug = PlugDescriptor("coatNormalZ")
	coatNormal_ : CoatNormalPlug = PlugDescriptor("coatNormal")
	coatRotation_ : CoatRotationPlug = PlugDescriptor("coatRotation")
	coatRoughness_ : CoatRoughnessPlug = PlugDescriptor("coatRoughness")
	diffuseRoughness_ : DiffuseRoughnessPlug = PlugDescriptor("diffuseRoughness")
	emission_ : EmissionPlug = PlugDescriptor("emission")
	emissionColorB_ : EmissionColorBPlug = PlugDescriptor("emissionColorB")
	emissionColorG_ : EmissionColorGPlug = PlugDescriptor("emissionColorG")
	emissionColorR_ : EmissionColorRPlug = PlugDescriptor("emissionColorR")
	emissionColor_ : EmissionColorPlug = PlugDescriptor("emissionColor")
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
	metalness_ : MetalnessPlug = PlugDescriptor("metalness")
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	normalCamera_ : NormalCameraPlug = PlugDescriptor("normalCamera")
	objectId_ : ObjectIdPlug = PlugDescriptor("objectId")
	opacityB_ : OpacityBPlug = PlugDescriptor("opacityB")
	opacityG_ : OpacityGPlug = PlugDescriptor("opacityG")
	opacityR_ : OpacityRPlug = PlugDescriptor("opacityR")
	opacity_ : OpacityPlug = PlugDescriptor("opacity")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
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
	sheen_ : SheenPlug = PlugDescriptor("sheen")
	sheenColorB_ : SheenColorBPlug = PlugDescriptor("sheenColorB")
	sheenColorG_ : SheenColorGPlug = PlugDescriptor("sheenColorG")
	sheenColorR_ : SheenColorRPlug = PlugDescriptor("sheenColorR")
	sheenColor_ : SheenColorPlug = PlugDescriptor("sheenColor")
	sheenRoughness_ : SheenRoughnessPlug = PlugDescriptor("sheenRoughness")
	specular_ : SpecularPlug = PlugDescriptor("specular")
	specularAnisotropy_ : SpecularAnisotropyPlug = PlugDescriptor("specularAnisotropy")
	specularColorB_ : SpecularColorBPlug = PlugDescriptor("specularColorB")
	specularColorG_ : SpecularColorGPlug = PlugDescriptor("specularColorG")
	specularColorR_ : SpecularColorRPlug = PlugDescriptor("specularColorR")
	specularColor_ : SpecularColorPlug = PlugDescriptor("specularColor")
	specularIOR_ : SpecularIORPlug = PlugDescriptor("specularIOR")
	specularRotation_ : SpecularRotationPlug = PlugDescriptor("specularRotation")
	specularRoughness_ : SpecularRoughnessPlug = PlugDescriptor("specularRoughness")
	subsurface_ : SubsurfacePlug = PlugDescriptor("subsurface")
	subsurfaceAnisotropy_ : SubsurfaceAnisotropyPlug = PlugDescriptor("subsurfaceAnisotropy")
	subsurfaceColorB_ : SubsurfaceColorBPlug = PlugDescriptor("subsurfaceColorB")
	subsurfaceColorG_ : SubsurfaceColorGPlug = PlugDescriptor("subsurfaceColorG")
	subsurfaceColorR_ : SubsurfaceColorRPlug = PlugDescriptor("subsurfaceColorR")
	subsurfaceColor_ : SubsurfaceColorPlug = PlugDescriptor("subsurfaceColor")
	subsurfaceRadiusB_ : SubsurfaceRadiusBPlug = PlugDescriptor("subsurfaceRadiusB")
	subsurfaceRadiusG_ : SubsurfaceRadiusGPlug = PlugDescriptor("subsurfaceRadiusG")
	subsurfaceRadiusR_ : SubsurfaceRadiusRPlug = PlugDescriptor("subsurfaceRadiusR")
	subsurfaceRadius_ : SubsurfaceRadiusPlug = PlugDescriptor("subsurfaceRadius")
	subsurfaceScale_ : SubsurfaceScalePlug = PlugDescriptor("subsurfaceScale")
	tangentUCameraX_ : TangentUCameraXPlug = PlugDescriptor("tangentUCameraX")
	tangentUCameraY_ : TangentUCameraYPlug = PlugDescriptor("tangentUCameraY")
	tangentUCameraZ_ : TangentUCameraZPlug = PlugDescriptor("tangentUCameraZ")
	tangentUCamera_ : TangentUCameraPlug = PlugDescriptor("tangentUCamera")
	thinFilmIOR_ : ThinFilmIORPlug = PlugDescriptor("thinFilmIOR")
	thinFilmThickness_ : ThinFilmThicknessPlug = PlugDescriptor("thinFilmThickness")
	thinWalled_ : ThinWalledPlug = PlugDescriptor("thinWalled")
	transmission_ : TransmissionPlug = PlugDescriptor("transmission")
	transmissionColorB_ : TransmissionColorBPlug = PlugDescriptor("transmissionColorB")
	transmissionColorG_ : TransmissionColorGPlug = PlugDescriptor("transmissionColorG")
	transmissionColorR_ : TransmissionColorRPlug = PlugDescriptor("transmissionColorR")
	transmissionColor_ : TransmissionColorPlug = PlugDescriptor("transmissionColor")
	transmissionDepth_ : TransmissionDepthPlug = PlugDescriptor("transmissionDepth")
	transmissionDispersion_ : TransmissionDispersionPlug = PlugDescriptor("transmissionDispersion")
	transmissionExtraRoughness_ : TransmissionExtraRoughnessPlug = PlugDescriptor("transmissionExtraRoughness")
	transmissionScatterB_ : TransmissionScatterBPlug = PlugDescriptor("transmissionScatterB")
	transmissionScatterG_ : TransmissionScatterGPlug = PlugDescriptor("transmissionScatterG")
	transmissionScatterR_ : TransmissionScatterRPlug = PlugDescriptor("transmissionScatterR")
	transmissionScatter_ : TransmissionScatterPlug = PlugDescriptor("transmissionScatter")
	transmissionScatterAnisotropy_ : TransmissionScatterAnisotropyPlug = PlugDescriptor("transmissionScatterAnisotropy")
	triangleNormalCameraX_ : TriangleNormalCameraXPlug = PlugDescriptor("triangleNormalCameraX")
	triangleNormalCameraY_ : TriangleNormalCameraYPlug = PlugDescriptor("triangleNormalCameraY")
	triangleNormalCameraZ_ : TriangleNormalCameraZPlug = PlugDescriptor("triangleNormalCameraZ")
	triangleNormalCamera_ : TriangleNormalCameraPlug = PlugDescriptor("triangleNormalCamera")

	# node attributes

	typeName = "standardSurface"
	apiTypeInt = 377
	apiTypeStr = "kStandardSurface"
	typeIdInt = 1398031443
	MFnCls = om.MFnDependencyNode
	pass

