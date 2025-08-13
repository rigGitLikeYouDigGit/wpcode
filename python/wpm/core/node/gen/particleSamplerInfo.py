

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
class AccelerationXPlug(Plug):
	parent : AccelerationPlug = PlugDescriptor("acceleration")
	node : ParticleSamplerInfo = None
	pass
class AccelerationYPlug(Plug):
	parent : AccelerationPlug = PlugDescriptor("acceleration")
	node : ParticleSamplerInfo = None
	pass
class AccelerationZPlug(Plug):
	parent : AccelerationPlug = PlugDescriptor("acceleration")
	node : ParticleSamplerInfo = None
	pass
class AccelerationPlug(Plug):
	accelerationX_ : AccelerationXPlug = PlugDescriptor("accelerationX")
	accx_ : AccelerationXPlug = PlugDescriptor("accelerationX")
	accelerationY_ : AccelerationYPlug = PlugDescriptor("accelerationY")
	accy_ : AccelerationYPlug = PlugDescriptor("accelerationY")
	accelerationZ_ : AccelerationZPlug = PlugDescriptor("accelerationZ")
	accz_ : AccelerationZPlug = PlugDescriptor("accelerationZ")
	node : ParticleSamplerInfo = None
	pass
class AgePlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class AgeNormalizedPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class BirthPositionXPlug(Plug):
	parent : BirthPositionPlug = PlugDescriptor("birthPosition")
	node : ParticleSamplerInfo = None
	pass
class BirthPositionYPlug(Plug):
	parent : BirthPositionPlug = PlugDescriptor("birthPosition")
	node : ParticleSamplerInfo = None
	pass
class BirthPositionZPlug(Plug):
	parent : BirthPositionPlug = PlugDescriptor("birthPosition")
	node : ParticleSamplerInfo = None
	pass
class BirthPositionPlug(Plug):
	birthPositionX_ : BirthPositionXPlug = PlugDescriptor("birthPositionX")
	bpox_ : BirthPositionXPlug = PlugDescriptor("birthPositionX")
	birthPositionY_ : BirthPositionYPlug = PlugDescriptor("birthPositionY")
	bpoy_ : BirthPositionYPlug = PlugDescriptor("birthPositionY")
	birthPositionZ_ : BirthPositionZPlug = PlugDescriptor("birthPositionZ")
	bpoz_ : BirthPositionZPlug = PlugDescriptor("birthPositionZ")
	node : ParticleSamplerInfo = None
	pass
class BirthTimePlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class BirthWorldPositionXPlug(Plug):
	parent : BirthWorldPositionPlug = PlugDescriptor("birthWorldPosition")
	node : ParticleSamplerInfo = None
	pass
class BirthWorldPositionYPlug(Plug):
	parent : BirthWorldPositionPlug = PlugDescriptor("birthWorldPosition")
	node : ParticleSamplerInfo = None
	pass
class BirthWorldPositionZPlug(Plug):
	parent : BirthWorldPositionPlug = PlugDescriptor("birthWorldPosition")
	node : ParticleSamplerInfo = None
	pass
class BirthWorldPositionPlug(Plug):
	birthWorldPositionX_ : BirthWorldPositionXPlug = PlugDescriptor("birthWorldPositionX")
	bwpx_ : BirthWorldPositionXPlug = PlugDescriptor("birthWorldPositionX")
	birthWorldPositionY_ : BirthWorldPositionYPlug = PlugDescriptor("birthWorldPositionY")
	bwpy_ : BirthWorldPositionYPlug = PlugDescriptor("birthWorldPositionY")
	birthWorldPositionZ_ : BirthWorldPositionZPlug = PlugDescriptor("birthWorldPositionZ")
	bwpz_ : BirthWorldPositionZPlug = PlugDescriptor("birthWorldPositionZ")
	node : ParticleSamplerInfo = None
	pass
class CollisionUPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class CollisionVPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class ColorBluePlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class ColorGreenPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class ColorRedPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class FinalLifespanPPPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class ForceXPlug(Plug):
	parent : ForcePlug = PlugDescriptor("force")
	node : ParticleSamplerInfo = None
	pass
class ForceYPlug(Plug):
	parent : ForcePlug = PlugDescriptor("force")
	node : ParticleSamplerInfo = None
	pass
class ForceZPlug(Plug):
	parent : ForcePlug = PlugDescriptor("force")
	node : ParticleSamplerInfo = None
	pass
class ForcePlug(Plug):
	forceX_ : ForceXPlug = PlugDescriptor("forceX")
	frx_ : ForceXPlug = PlugDescriptor("forceX")
	forceY_ : ForceYPlug = PlugDescriptor("forceY")
	fry_ : ForceYPlug = PlugDescriptor("forceY")
	forceZ_ : ForceZPlug = PlugDescriptor("forceZ")
	frz_ : ForceZPlug = PlugDescriptor("forceZ")
	node : ParticleSamplerInfo = None
	pass
class IncandescenceBPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : ParticleSamplerInfo = None
	pass
class IncandescenceGPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : ParticleSamplerInfo = None
	pass
class IncandescenceRPlug(Plug):
	parent : IncandescencePlug = PlugDescriptor("incandescence")
	node : ParticleSamplerInfo = None
	pass
class IncandescencePlug(Plug):
	incandescenceB_ : IncandescenceBPlug = PlugDescriptor("incandescenceB")
	inb_ : IncandescenceBPlug = PlugDescriptor("incandescenceB")
	incandescenceG_ : IncandescenceGPlug = PlugDescriptor("incandescenceG")
	ing_ : IncandescenceGPlug = PlugDescriptor("incandescenceG")
	incandescenceR_ : IncandescenceRPlug = PlugDescriptor("incandescenceR")
	inr_ : IncandescenceRPlug = PlugDescriptor("incandescenceR")
	node : ParticleSamplerInfo = None
	pass
class IncandescencePPBPlug(Plug):
	parent : IncandescencePPPlug = PlugDescriptor("incandescencePP")
	node : ParticleSamplerInfo = None
	pass
class IncandescencePPGPlug(Plug):
	parent : IncandescencePPPlug = PlugDescriptor("incandescencePP")
	node : ParticleSamplerInfo = None
	pass
class IncandescencePPRPlug(Plug):
	parent : IncandescencePPPlug = PlugDescriptor("incandescencePP")
	node : ParticleSamplerInfo = None
	pass
class IncandescencePPPlug(Plug):
	incandescencePPB_ : IncandescencePPBPlug = PlugDescriptor("incandescencePPB")
	ippb_ : IncandescencePPBPlug = PlugDescriptor("incandescencePPB")
	incandescencePPG_ : IncandescencePPGPlug = PlugDescriptor("incandescencePPG")
	ippg_ : IncandescencePPGPlug = PlugDescriptor("incandescencePPG")
	incandescencePPR_ : IncandescencePPRPlug = PlugDescriptor("incandescencePPR")
	ippr_ : IncandescencePPRPlug = PlugDescriptor("incandescencePPR")
	node : ParticleSamplerInfo = None
	pass
class InverseOutUvPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class LifespanPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class LifespanPPPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class NormalizationMethodPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class NormalizationValuePlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class ObjectTypePlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class OpacityPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class OpacityPPPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : ParticleSamplerInfo = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : ParticleSamplerInfo = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : ParticleSamplerInfo = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : ParticleSamplerInfo = None
	pass
class OutIncandescenceBPlug(Plug):
	parent : OutIncandescencePlug = PlugDescriptor("outIncandescence")
	node : ParticleSamplerInfo = None
	pass
class OutIncandescenceGPlug(Plug):
	parent : OutIncandescencePlug = PlugDescriptor("outIncandescence")
	node : ParticleSamplerInfo = None
	pass
class OutIncandescenceRPlug(Plug):
	parent : OutIncandescencePlug = PlugDescriptor("outIncandescence")
	node : ParticleSamplerInfo = None
	pass
class OutIncandescencePlug(Plug):
	outIncandescenceB_ : OutIncandescenceBPlug = PlugDescriptor("outIncandescenceB")
	oicb_ : OutIncandescenceBPlug = PlugDescriptor("outIncandescenceB")
	outIncandescenceG_ : OutIncandescenceGPlug = PlugDescriptor("outIncandescenceG")
	oicg_ : OutIncandescenceGPlug = PlugDescriptor("outIncandescenceG")
	outIncandescenceR_ : OutIncandescenceRPlug = PlugDescriptor("outIncandescenceR")
	oicr_ : OutIncandescenceRPlug = PlugDescriptor("outIncandescenceR")
	node : ParticleSamplerInfo = None
	pass
class OutTransparencyBPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : ParticleSamplerInfo = None
	pass
class OutTransparencyGPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : ParticleSamplerInfo = None
	pass
class OutTransparencyRPlug(Plug):
	parent : OutTransparencyPlug = PlugDescriptor("outTransparency")
	node : ParticleSamplerInfo = None
	pass
class OutTransparencyPlug(Plug):
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	otb_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	otg_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	otr_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	node : ParticleSamplerInfo = None
	pass
class OutUCoordPlug(Plug):
	parent : OutUvCoordPlug = PlugDescriptor("outUvCoord")
	node : ParticleSamplerInfo = None
	pass
class OutVCoordPlug(Plug):
	parent : OutUvCoordPlug = PlugDescriptor("outUvCoord")
	node : ParticleSamplerInfo = None
	pass
class OutUvCoordPlug(Plug):
	outUCoord_ : OutUCoordPlug = PlugDescriptor("outUCoord")
	ouc_ : OutUCoordPlug = PlugDescriptor("outUCoord")
	outVCoord_ : OutVCoordPlug = PlugDescriptor("outVCoord")
	ovc_ : OutVCoordPlug = PlugDescriptor("outVCoord")
	node : ParticleSamplerInfo = None
	pass
class OutUvTypePlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class ParentUPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class ParentVPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class ParticleAgePlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class ParticleAttrArrayPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class ParticleColorBPlug(Plug):
	parent : ParticleColorPlug = PlugDescriptor("particleColor")
	node : ParticleSamplerInfo = None
	pass
class ParticleColorGPlug(Plug):
	parent : ParticleColorPlug = PlugDescriptor("particleColor")
	node : ParticleSamplerInfo = None
	pass
class ParticleColorRPlug(Plug):
	parent : ParticleColorPlug = PlugDescriptor("particleColor")
	node : ParticleSamplerInfo = None
	pass
class ParticleColorPlug(Plug):
	particleColorB_ : ParticleColorBPlug = PlugDescriptor("particleColorB")
	pcb_ : ParticleColorBPlug = PlugDescriptor("particleColorB")
	particleColorG_ : ParticleColorGPlug = PlugDescriptor("particleColorG")
	pcg_ : ParticleColorGPlug = PlugDescriptor("particleColorG")
	particleColorR_ : ParticleColorRPlug = PlugDescriptor("particleColorR")
	pcr_ : ParticleColorRPlug = PlugDescriptor("particleColorR")
	node : ParticleSamplerInfo = None
	pass
class ParticleIdPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class ParticleIncandescenceBPlug(Plug):
	parent : ParticleIncandescencePlug = PlugDescriptor("particleIncandescence")
	node : ParticleSamplerInfo = None
	pass
class ParticleIncandescenceGPlug(Plug):
	parent : ParticleIncandescencePlug = PlugDescriptor("particleIncandescence")
	node : ParticleSamplerInfo = None
	pass
class ParticleIncandescenceRPlug(Plug):
	parent : ParticleIncandescencePlug = PlugDescriptor("particleIncandescence")
	node : ParticleSamplerInfo = None
	pass
class ParticleIncandescencePlug(Plug):
	particleIncandescenceB_ : ParticleIncandescenceBPlug = PlugDescriptor("particleIncandescenceB")
	pib_ : ParticleIncandescenceBPlug = PlugDescriptor("particleIncandescenceB")
	particleIncandescenceG_ : ParticleIncandescenceGPlug = PlugDescriptor("particleIncandescenceG")
	pig_ : ParticleIncandescenceGPlug = PlugDescriptor("particleIncandescenceG")
	particleIncandescenceR_ : ParticleIncandescenceRPlug = PlugDescriptor("particleIncandescenceR")
	pir_ : ParticleIncandescenceRPlug = PlugDescriptor("particleIncandescenceR")
	node : ParticleSamplerInfo = None
	pass
class ParticleLifespanPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class ParticleOrderPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class ParticleTransparencyBPlug(Plug):
	parent : ParticleTransparencyPlug = PlugDescriptor("particleTransparency")
	node : ParticleSamplerInfo = None
	pass
class ParticleTransparencyGPlug(Plug):
	parent : ParticleTransparencyPlug = PlugDescriptor("particleTransparency")
	node : ParticleSamplerInfo = None
	pass
class ParticleTransparencyRPlug(Plug):
	parent : ParticleTransparencyPlug = PlugDescriptor("particleTransparency")
	node : ParticleSamplerInfo = None
	pass
class ParticleTransparencyPlug(Plug):
	particleTransparencyB_ : ParticleTransparencyBPlug = PlugDescriptor("particleTransparencyB")
	ptb_ : ParticleTransparencyBPlug = PlugDescriptor("particleTransparencyB")
	particleTransparencyG_ : ParticleTransparencyGPlug = PlugDescriptor("particleTransparencyG")
	ptg_ : ParticleTransparencyGPlug = PlugDescriptor("particleTransparencyG")
	particleTransparencyR_ : ParticleTransparencyRPlug = PlugDescriptor("particleTransparencyR")
	ptr_ : ParticleTransparencyRPlug = PlugDescriptor("particleTransparencyR")
	node : ParticleSamplerInfo = None
	pass
class PositionXPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : ParticleSamplerInfo = None
	pass
class PositionYPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : ParticleSamplerInfo = None
	pass
class PositionZPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : ParticleSamplerInfo = None
	pass
class PositionPlug(Plug):
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	posx_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	posy_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	posz_ : PositionZPlug = PlugDescriptor("positionZ")
	node : ParticleSamplerInfo = None
	pass
class RadiusPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class RadiusPPPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class BPPPlug(Plug):
	parent : RgbPPPlug = PlugDescriptor("rgbPP")
	node : ParticleSamplerInfo = None
	pass
class GPPPlug(Plug):
	parent : RgbPPPlug = PlugDescriptor("rgbPP")
	node : ParticleSamplerInfo = None
	pass
class RPPPlug(Plug):
	parent : RgbPPPlug = PlugDescriptor("rgbPP")
	node : ParticleSamplerInfo = None
	pass
class RgbPPPlug(Plug):
	bPP_ : BPPPlug = PlugDescriptor("bPP")
	bpp_ : BPPPlug = PlugDescriptor("bPP")
	gPP_ : GPPPlug = PlugDescriptor("gPP")
	gpp_ : GPPPlug = PlugDescriptor("gPP")
	rPP_ : RPPPlug = PlugDescriptor("rPP")
	rpp_ : RPPPlug = PlugDescriptor("rPP")
	node : ParticleSamplerInfo = None
	pass
class UserScalar1PPPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class UserScalar2PPPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class UserScalar3PPPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class UserScalar4PPPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class UserScalar5PPPlug(Plug):
	node : ParticleSamplerInfo = None
	pass
class UserVector1PPXPlug(Plug):
	parent : UserVector1PPPlug = PlugDescriptor("userVector1PP")
	node : ParticleSamplerInfo = None
	pass
class UserVector1PPYPlug(Plug):
	parent : UserVector1PPPlug = PlugDescriptor("userVector1PP")
	node : ParticleSamplerInfo = None
	pass
class UserVector1PPZPlug(Plug):
	parent : UserVector1PPPlug = PlugDescriptor("userVector1PP")
	node : ParticleSamplerInfo = None
	pass
class UserVector1PPPlug(Plug):
	userVector1PPX_ : UserVector1PPXPlug = PlugDescriptor("userVector1PPX")
	uv1x_ : UserVector1PPXPlug = PlugDescriptor("userVector1PPX")
	userVector1PPY_ : UserVector1PPYPlug = PlugDescriptor("userVector1PPY")
	uv1y_ : UserVector1PPYPlug = PlugDescriptor("userVector1PPY")
	userVector1PPZ_ : UserVector1PPZPlug = PlugDescriptor("userVector1PPZ")
	uv1z_ : UserVector1PPZPlug = PlugDescriptor("userVector1PPZ")
	node : ParticleSamplerInfo = None
	pass
class UserVector2PPXPlug(Plug):
	parent : UserVector2PPPlug = PlugDescriptor("userVector2PP")
	node : ParticleSamplerInfo = None
	pass
class UserVector2PPYPlug(Plug):
	parent : UserVector2PPPlug = PlugDescriptor("userVector2PP")
	node : ParticleSamplerInfo = None
	pass
class UserVector2PPZPlug(Plug):
	parent : UserVector2PPPlug = PlugDescriptor("userVector2PP")
	node : ParticleSamplerInfo = None
	pass
class UserVector2PPPlug(Plug):
	userVector2PPX_ : UserVector2PPXPlug = PlugDescriptor("userVector2PPX")
	uv2x_ : UserVector2PPXPlug = PlugDescriptor("userVector2PPX")
	userVector2PPY_ : UserVector2PPYPlug = PlugDescriptor("userVector2PPY")
	uv2y_ : UserVector2PPYPlug = PlugDescriptor("userVector2PPY")
	userVector2PPZ_ : UserVector2PPZPlug = PlugDescriptor("userVector2PPZ")
	uv2z_ : UserVector2PPZPlug = PlugDescriptor("userVector2PPZ")
	node : ParticleSamplerInfo = None
	pass
class UserVector3PPXPlug(Plug):
	parent : UserVector3PPPlug = PlugDescriptor("userVector3PP")
	node : ParticleSamplerInfo = None
	pass
class UserVector3PPYPlug(Plug):
	parent : UserVector3PPPlug = PlugDescriptor("userVector3PP")
	node : ParticleSamplerInfo = None
	pass
class UserVector3PPZPlug(Plug):
	parent : UserVector3PPPlug = PlugDescriptor("userVector3PP")
	node : ParticleSamplerInfo = None
	pass
class UserVector3PPPlug(Plug):
	userVector3PPX_ : UserVector3PPXPlug = PlugDescriptor("userVector3PPX")
	uv3x_ : UserVector3PPXPlug = PlugDescriptor("userVector3PPX")
	userVector3PPY_ : UserVector3PPYPlug = PlugDescriptor("userVector3PPY")
	uv3y_ : UserVector3PPYPlug = PlugDescriptor("userVector3PPY")
	userVector3PPZ_ : UserVector3PPZPlug = PlugDescriptor("userVector3PPZ")
	uv3z_ : UserVector3PPZPlug = PlugDescriptor("userVector3PPZ")
	node : ParticleSamplerInfo = None
	pass
class UserVector4PPXPlug(Plug):
	parent : UserVector4PPPlug = PlugDescriptor("userVector4PP")
	node : ParticleSamplerInfo = None
	pass
class UserVector4PPYPlug(Plug):
	parent : UserVector4PPPlug = PlugDescriptor("userVector4PP")
	node : ParticleSamplerInfo = None
	pass
class UserVector4PPZPlug(Plug):
	parent : UserVector4PPPlug = PlugDescriptor("userVector4PP")
	node : ParticleSamplerInfo = None
	pass
class UserVector4PPPlug(Plug):
	userVector4PPX_ : UserVector4PPXPlug = PlugDescriptor("userVector4PPX")
	uv4x_ : UserVector4PPXPlug = PlugDescriptor("userVector4PPX")
	userVector4PPY_ : UserVector4PPYPlug = PlugDescriptor("userVector4PPY")
	uv4y_ : UserVector4PPYPlug = PlugDescriptor("userVector4PPY")
	userVector4PPZ_ : UserVector4PPZPlug = PlugDescriptor("userVector4PPZ")
	uv4z_ : UserVector4PPZPlug = PlugDescriptor("userVector4PPZ")
	node : ParticleSamplerInfo = None
	pass
class UserVector5PPXPlug(Plug):
	parent : UserVector5PPPlug = PlugDescriptor("userVector5PP")
	node : ParticleSamplerInfo = None
	pass
class UserVector5PPYPlug(Plug):
	parent : UserVector5PPPlug = PlugDescriptor("userVector5PP")
	node : ParticleSamplerInfo = None
	pass
class UserVector5PPZPlug(Plug):
	parent : UserVector5PPPlug = PlugDescriptor("userVector5PP")
	node : ParticleSamplerInfo = None
	pass
class UserVector5PPPlug(Plug):
	userVector5PPX_ : UserVector5PPXPlug = PlugDescriptor("userVector5PPX")
	uv5x_ : UserVector5PPXPlug = PlugDescriptor("userVector5PPX")
	userVector5PPY_ : UserVector5PPYPlug = PlugDescriptor("userVector5PPY")
	uv5y_ : UserVector5PPYPlug = PlugDescriptor("userVector5PPY")
	userVector5PPZ_ : UserVector5PPZPlug = PlugDescriptor("userVector5PPZ")
	uv5z_ : UserVector5PPZPlug = PlugDescriptor("userVector5PPZ")
	node : ParticleSamplerInfo = None
	pass
class VelocityXPlug(Plug):
	parent : VelocityPlug = PlugDescriptor("velocity")
	node : ParticleSamplerInfo = None
	pass
class VelocityYPlug(Plug):
	parent : VelocityPlug = PlugDescriptor("velocity")
	node : ParticleSamplerInfo = None
	pass
class VelocityZPlug(Plug):
	parent : VelocityPlug = PlugDescriptor("velocity")
	node : ParticleSamplerInfo = None
	pass
class VelocityPlug(Plug):
	velocityX_ : VelocityXPlug = PlugDescriptor("velocityX")
	velx_ : VelocityXPlug = PlugDescriptor("velocityX")
	velocityY_ : VelocityYPlug = PlugDescriptor("velocityY")
	vely_ : VelocityYPlug = PlugDescriptor("velocityY")
	velocityZ_ : VelocityZPlug = PlugDescriptor("velocityZ")
	velz_ : VelocityZPlug = PlugDescriptor("velocityZ")
	node : ParticleSamplerInfo = None
	pass
class WorldPositionXPlug(Plug):
	parent : WorldPositionPlug = PlugDescriptor("worldPosition")
	node : ParticleSamplerInfo = None
	pass
class WorldPositionYPlug(Plug):
	parent : WorldPositionPlug = PlugDescriptor("worldPosition")
	node : ParticleSamplerInfo = None
	pass
class WorldPositionZPlug(Plug):
	parent : WorldPositionPlug = PlugDescriptor("worldPosition")
	node : ParticleSamplerInfo = None
	pass
class WorldPositionPlug(Plug):
	worldPositionX_ : WorldPositionXPlug = PlugDescriptor("worldPositionX")
	wpsx_ : WorldPositionXPlug = PlugDescriptor("worldPositionX")
	worldPositionY_ : WorldPositionYPlug = PlugDescriptor("worldPositionY")
	wpsy_ : WorldPositionYPlug = PlugDescriptor("worldPositionY")
	worldPositionZ_ : WorldPositionZPlug = PlugDescriptor("worldPositionZ")
	wpsz_ : WorldPositionZPlug = PlugDescriptor("worldPositionZ")
	node : ParticleSamplerInfo = None
	pass
class WorldVelocityXPlug(Plug):
	parent : WorldVelocityPlug = PlugDescriptor("worldVelocity")
	node : ParticleSamplerInfo = None
	pass
class WorldVelocityYPlug(Plug):
	parent : WorldVelocityPlug = PlugDescriptor("worldVelocity")
	node : ParticleSamplerInfo = None
	pass
class WorldVelocityZPlug(Plug):
	parent : WorldVelocityPlug = PlugDescriptor("worldVelocity")
	node : ParticleSamplerInfo = None
	pass
class WorldVelocityPlug(Plug):
	worldVelocityX_ : WorldVelocityXPlug = PlugDescriptor("worldVelocityX")
	wvlx_ : WorldVelocityXPlug = PlugDescriptor("worldVelocityX")
	worldVelocityY_ : WorldVelocityYPlug = PlugDescriptor("worldVelocityY")
	wvly_ : WorldVelocityYPlug = PlugDescriptor("worldVelocityY")
	worldVelocityZ_ : WorldVelocityZPlug = PlugDescriptor("worldVelocityZ")
	wvlz_ : WorldVelocityZPlug = PlugDescriptor("worldVelocityZ")
	node : ParticleSamplerInfo = None
	pass
# endregion


# define node class
class ParticleSamplerInfo(ShadingDependNode):
	accelerationX_ : AccelerationXPlug = PlugDescriptor("accelerationX")
	accelerationY_ : AccelerationYPlug = PlugDescriptor("accelerationY")
	accelerationZ_ : AccelerationZPlug = PlugDescriptor("accelerationZ")
	acceleration_ : AccelerationPlug = PlugDescriptor("acceleration")
	age_ : AgePlug = PlugDescriptor("age")
	ageNormalized_ : AgeNormalizedPlug = PlugDescriptor("ageNormalized")
	birthPositionX_ : BirthPositionXPlug = PlugDescriptor("birthPositionX")
	birthPositionY_ : BirthPositionYPlug = PlugDescriptor("birthPositionY")
	birthPositionZ_ : BirthPositionZPlug = PlugDescriptor("birthPositionZ")
	birthPosition_ : BirthPositionPlug = PlugDescriptor("birthPosition")
	birthTime_ : BirthTimePlug = PlugDescriptor("birthTime")
	birthWorldPositionX_ : BirthWorldPositionXPlug = PlugDescriptor("birthWorldPositionX")
	birthWorldPositionY_ : BirthWorldPositionYPlug = PlugDescriptor("birthWorldPositionY")
	birthWorldPositionZ_ : BirthWorldPositionZPlug = PlugDescriptor("birthWorldPositionZ")
	birthWorldPosition_ : BirthWorldPositionPlug = PlugDescriptor("birthWorldPosition")
	collisionU_ : CollisionUPlug = PlugDescriptor("collisionU")
	collisionV_ : CollisionVPlug = PlugDescriptor("collisionV")
	colorBlue_ : ColorBluePlug = PlugDescriptor("colorBlue")
	colorGreen_ : ColorGreenPlug = PlugDescriptor("colorGreen")
	colorRed_ : ColorRedPlug = PlugDescriptor("colorRed")
	finalLifespanPP_ : FinalLifespanPPPlug = PlugDescriptor("finalLifespanPP")
	forceX_ : ForceXPlug = PlugDescriptor("forceX")
	forceY_ : ForceYPlug = PlugDescriptor("forceY")
	forceZ_ : ForceZPlug = PlugDescriptor("forceZ")
	force_ : ForcePlug = PlugDescriptor("force")
	incandescenceB_ : IncandescenceBPlug = PlugDescriptor("incandescenceB")
	incandescenceG_ : IncandescenceGPlug = PlugDescriptor("incandescenceG")
	incandescenceR_ : IncandescenceRPlug = PlugDescriptor("incandescenceR")
	incandescence_ : IncandescencePlug = PlugDescriptor("incandescence")
	incandescencePPB_ : IncandescencePPBPlug = PlugDescriptor("incandescencePPB")
	incandescencePPG_ : IncandescencePPGPlug = PlugDescriptor("incandescencePPG")
	incandescencePPR_ : IncandescencePPRPlug = PlugDescriptor("incandescencePPR")
	incandescencePP_ : IncandescencePPPlug = PlugDescriptor("incandescencePP")
	inverseOutUv_ : InverseOutUvPlug = PlugDescriptor("inverseOutUv")
	lifespan_ : LifespanPlug = PlugDescriptor("lifespan")
	lifespanPP_ : LifespanPPPlug = PlugDescriptor("lifespanPP")
	normalizationMethod_ : NormalizationMethodPlug = PlugDescriptor("normalizationMethod")
	normalizationValue_ : NormalizationValuePlug = PlugDescriptor("normalizationValue")
	objectType_ : ObjectTypePlug = PlugDescriptor("objectType")
	opacity_ : OpacityPlug = PlugDescriptor("opacity")
	opacityPP_ : OpacityPPPlug = PlugDescriptor("opacityPP")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	outIncandescenceB_ : OutIncandescenceBPlug = PlugDescriptor("outIncandescenceB")
	outIncandescenceG_ : OutIncandescenceGPlug = PlugDescriptor("outIncandescenceG")
	outIncandescenceR_ : OutIncandescenceRPlug = PlugDescriptor("outIncandescenceR")
	outIncandescence_ : OutIncandescencePlug = PlugDescriptor("outIncandescence")
	outTransparencyB_ : OutTransparencyBPlug = PlugDescriptor("outTransparencyB")
	outTransparencyG_ : OutTransparencyGPlug = PlugDescriptor("outTransparencyG")
	outTransparencyR_ : OutTransparencyRPlug = PlugDescriptor("outTransparencyR")
	outTransparency_ : OutTransparencyPlug = PlugDescriptor("outTransparency")
	outUCoord_ : OutUCoordPlug = PlugDescriptor("outUCoord")
	outVCoord_ : OutVCoordPlug = PlugDescriptor("outVCoord")
	outUvCoord_ : OutUvCoordPlug = PlugDescriptor("outUvCoord")
	outUvType_ : OutUvTypePlug = PlugDescriptor("outUvType")
	parentU_ : ParentUPlug = PlugDescriptor("parentU")
	parentV_ : ParentVPlug = PlugDescriptor("parentV")
	particleAge_ : ParticleAgePlug = PlugDescriptor("particleAge")
	particleAttrArray_ : ParticleAttrArrayPlug = PlugDescriptor("particleAttrArray")
	particleColorB_ : ParticleColorBPlug = PlugDescriptor("particleColorB")
	particleColorG_ : ParticleColorGPlug = PlugDescriptor("particleColorG")
	particleColorR_ : ParticleColorRPlug = PlugDescriptor("particleColorR")
	particleColor_ : ParticleColorPlug = PlugDescriptor("particleColor")
	particleId_ : ParticleIdPlug = PlugDescriptor("particleId")
	particleIncandescenceB_ : ParticleIncandescenceBPlug = PlugDescriptor("particleIncandescenceB")
	particleIncandescenceG_ : ParticleIncandescenceGPlug = PlugDescriptor("particleIncandescenceG")
	particleIncandescenceR_ : ParticleIncandescenceRPlug = PlugDescriptor("particleIncandescenceR")
	particleIncandescence_ : ParticleIncandescencePlug = PlugDescriptor("particleIncandescence")
	particleLifespan_ : ParticleLifespanPlug = PlugDescriptor("particleLifespan")
	particleOrder_ : ParticleOrderPlug = PlugDescriptor("particleOrder")
	particleTransparencyB_ : ParticleTransparencyBPlug = PlugDescriptor("particleTransparencyB")
	particleTransparencyG_ : ParticleTransparencyGPlug = PlugDescriptor("particleTransparencyG")
	particleTransparencyR_ : ParticleTransparencyRPlug = PlugDescriptor("particleTransparencyR")
	particleTransparency_ : ParticleTransparencyPlug = PlugDescriptor("particleTransparency")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	position_ : PositionPlug = PlugDescriptor("position")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	radiusPP_ : RadiusPPPlug = PlugDescriptor("radiusPP")
	bPP_ : BPPPlug = PlugDescriptor("bPP")
	gPP_ : GPPPlug = PlugDescriptor("gPP")
	rPP_ : RPPPlug = PlugDescriptor("rPP")
	rgbPP_ : RgbPPPlug = PlugDescriptor("rgbPP")
	userScalar1PP_ : UserScalar1PPPlug = PlugDescriptor("userScalar1PP")
	userScalar2PP_ : UserScalar2PPPlug = PlugDescriptor("userScalar2PP")
	userScalar3PP_ : UserScalar3PPPlug = PlugDescriptor("userScalar3PP")
	userScalar4PP_ : UserScalar4PPPlug = PlugDescriptor("userScalar4PP")
	userScalar5PP_ : UserScalar5PPPlug = PlugDescriptor("userScalar5PP")
	userVector1PPX_ : UserVector1PPXPlug = PlugDescriptor("userVector1PPX")
	userVector1PPY_ : UserVector1PPYPlug = PlugDescriptor("userVector1PPY")
	userVector1PPZ_ : UserVector1PPZPlug = PlugDescriptor("userVector1PPZ")
	userVector1PP_ : UserVector1PPPlug = PlugDescriptor("userVector1PP")
	userVector2PPX_ : UserVector2PPXPlug = PlugDescriptor("userVector2PPX")
	userVector2PPY_ : UserVector2PPYPlug = PlugDescriptor("userVector2PPY")
	userVector2PPZ_ : UserVector2PPZPlug = PlugDescriptor("userVector2PPZ")
	userVector2PP_ : UserVector2PPPlug = PlugDescriptor("userVector2PP")
	userVector3PPX_ : UserVector3PPXPlug = PlugDescriptor("userVector3PPX")
	userVector3PPY_ : UserVector3PPYPlug = PlugDescriptor("userVector3PPY")
	userVector3PPZ_ : UserVector3PPZPlug = PlugDescriptor("userVector3PPZ")
	userVector3PP_ : UserVector3PPPlug = PlugDescriptor("userVector3PP")
	userVector4PPX_ : UserVector4PPXPlug = PlugDescriptor("userVector4PPX")
	userVector4PPY_ : UserVector4PPYPlug = PlugDescriptor("userVector4PPY")
	userVector4PPZ_ : UserVector4PPZPlug = PlugDescriptor("userVector4PPZ")
	userVector4PP_ : UserVector4PPPlug = PlugDescriptor("userVector4PP")
	userVector5PPX_ : UserVector5PPXPlug = PlugDescriptor("userVector5PPX")
	userVector5PPY_ : UserVector5PPYPlug = PlugDescriptor("userVector5PPY")
	userVector5PPZ_ : UserVector5PPZPlug = PlugDescriptor("userVector5PPZ")
	userVector5PP_ : UserVector5PPPlug = PlugDescriptor("userVector5PP")
	velocityX_ : VelocityXPlug = PlugDescriptor("velocityX")
	velocityY_ : VelocityYPlug = PlugDescriptor("velocityY")
	velocityZ_ : VelocityZPlug = PlugDescriptor("velocityZ")
	velocity_ : VelocityPlug = PlugDescriptor("velocity")
	worldPositionX_ : WorldPositionXPlug = PlugDescriptor("worldPositionX")
	worldPositionY_ : WorldPositionYPlug = PlugDescriptor("worldPositionY")
	worldPositionZ_ : WorldPositionZPlug = PlugDescriptor("worldPositionZ")
	worldPosition_ : WorldPositionPlug = PlugDescriptor("worldPosition")
	worldVelocityX_ : WorldVelocityXPlug = PlugDescriptor("worldVelocityX")
	worldVelocityY_ : WorldVelocityYPlug = PlugDescriptor("worldVelocityY")
	worldVelocityZ_ : WorldVelocityZPlug = PlugDescriptor("worldVelocityZ")
	worldVelocity_ : WorldVelocityPlug = PlugDescriptor("worldVelocity")

	# node attributes

	typeName = "particleSamplerInfo"
	apiTypeInt = 806
	apiTypeStr = "kParticleSamplerInfo"
	typeIdInt = 1347635534
	MFnCls = om.MFnDependencyNode
	pass

