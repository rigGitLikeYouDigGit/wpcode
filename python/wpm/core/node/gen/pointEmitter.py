

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	DynBase = Catalogue.DynBase
else:
	from .. import retriever
	DynBase = retriever.getNodeCls("DynBase")
	assert DynBase

# add node doc



# region plug type defs
class AlongAxisPlug(Plug):
	node : PointEmitter = None
	pass
class AroundAxisPlug(Plug):
	node : PointEmitter = None
	pass
class AwayFromAxisPlug(Plug):
	node : PointEmitter = None
	pass
class AwayFromCenterPlug(Plug):
	node : PointEmitter = None
	pass
class CurrentTimePlug(Plug):
	node : PointEmitter = None
	pass
class CycleEmissionPlug(Plug):
	node : PointEmitter = None
	pass
class CycleIntervalPlug(Plug):
	node : PointEmitter = None
	pass
class DeltaTimePlug(Plug):
	node : PointEmitter = None
	pass
class DeltaTimeCyclePlug(Plug):
	node : PointEmitter = None
	pass
class DirectionXPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : PointEmitter = None
	pass
class DirectionYPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : PointEmitter = None
	pass
class DirectionZPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : PointEmitter = None
	pass
class DirectionPlug(Plug):
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	dx_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	dy_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	dz_ : DirectionZPlug = PlugDescriptor("directionZ")
	node : PointEmitter = None
	pass
class DirectionalSpeedPlug(Plug):
	node : PointEmitter = None
	pass
class DisplaySpeedPlug(Plug):
	node : PointEmitter = None
	pass
class EmitCountRemainderPlug(Plug):
	node : PointEmitter = None
	pass
class EmitFromDarkPlug(Plug):
	node : PointEmitter = None
	pass
class EmitterTypePlug(Plug):
	node : PointEmitter = None
	pass
class EnableTextureRatePlug(Plug):
	node : PointEmitter = None
	pass
class InheritColorPlug(Plug):
	node : PointEmitter = None
	pass
class InheritFactorPlug(Plug):
	node : PointEmitter = None
	pass
class InheritOpacityPlug(Plug):
	node : PointEmitter = None
	pass
class InvertOpacityPlug(Plug):
	node : PointEmitter = None
	pass
class IsFullPlug(Plug):
	node : PointEmitter = None
	pass
class MaxDistancePlug(Plug):
	node : PointEmitter = None
	pass
class MinDistancePlug(Plug):
	node : PointEmitter = None
	pass
class NeedParentUVPlug(Plug):
	node : PointEmitter = None
	pass
class NormalSpeedPlug(Plug):
	node : PointEmitter = None
	pass
class OutputPlug(Plug):
	node : PointEmitter = None
	pass
class ParentIdPlug(Plug):
	node : PointEmitter = None
	pass
class ParticleColorBPlug(Plug):
	parent : ParticleColorPlug = PlugDescriptor("particleColor")
	node : PointEmitter = None
	pass
class ParticleColorGPlug(Plug):
	parent : ParticleColorPlug = PlugDescriptor("particleColor")
	node : PointEmitter = None
	pass
class ParticleColorRPlug(Plug):
	parent : ParticleColorPlug = PlugDescriptor("particleColor")
	node : PointEmitter = None
	pass
class ParticleColorPlug(Plug):
	particleColorB_ : ParticleColorBPlug = PlugDescriptor("particleColorB")
	pcb_ : ParticleColorBPlug = PlugDescriptor("particleColorB")
	particleColorG_ : ParticleColorGPlug = PlugDescriptor("particleColorG")
	pcg_ : ParticleColorGPlug = PlugDescriptor("particleColorG")
	particleColorR_ : ParticleColorRPlug = PlugDescriptor("particleColorR")
	pcr_ : ParticleColorRPlug = PlugDescriptor("particleColorR")
	node : PointEmitter = None
	pass
class RandStateXPlug(Plug):
	parent : RandStatePlug = PlugDescriptor("randState")
	node : PointEmitter = None
	pass
class RandStateYPlug(Plug):
	parent : RandStatePlug = PlugDescriptor("randState")
	node : PointEmitter = None
	pass
class RandStateZPlug(Plug):
	parent : RandStatePlug = PlugDescriptor("randState")
	node : PointEmitter = None
	pass
class RandStatePlug(Plug):
	randStateX_ : RandStateXPlug = PlugDescriptor("randStateX")
	rstx_ : RandStateXPlug = PlugDescriptor("randStateX")
	randStateY_ : RandStateYPlug = PlugDescriptor("randStateY")
	rsty_ : RandStateYPlug = PlugDescriptor("randStateY")
	randStateZ_ : RandStateZPlug = PlugDescriptor("randStateZ")
	rstz_ : RandStateZPlug = PlugDescriptor("randStateZ")
	node : PointEmitter = None
	pass
class RandomDirectionPlug(Plug):
	node : PointEmitter = None
	pass
class RatePlug(Plug):
	node : PointEmitter = None
	pass
class RatePPPlug(Plug):
	node : PointEmitter = None
	pass
class ScaleRateByObjectSizePlug(Plug):
	node : PointEmitter = None
	pass
class ScaleRateBySpeedPlug(Plug):
	node : PointEmitter = None
	pass
class ScaleSpeedBySizePlug(Plug):
	node : PointEmitter = None
	pass
class SectionRadiusPlug(Plug):
	node : PointEmitter = None
	pass
class SeedPlug(Plug):
	node : PointEmitter = None
	pass
class SpeedPlug(Plug):
	node : PointEmitter = None
	pass
class SpeedRandomPlug(Plug):
	node : PointEmitter = None
	pass
class SpreadPlug(Plug):
	node : PointEmitter = None
	pass
class StartTimePlug(Plug):
	node : PointEmitter = None
	pass
class SweptGeometryPlug(Plug):
	node : PointEmitter = None
	pass
class TangentSpeedPlug(Plug):
	node : PointEmitter = None
	pass
class TextureRateBPlug(Plug):
	parent : TextureRatePlug = PlugDescriptor("textureRate")
	node : PointEmitter = None
	pass
class TextureRateGPlug(Plug):
	parent : TextureRatePlug = PlugDescriptor("textureRate")
	node : PointEmitter = None
	pass
class TextureRateRPlug(Plug):
	parent : TextureRatePlug = PlugDescriptor("textureRate")
	node : PointEmitter = None
	pass
class TextureRatePlug(Plug):
	textureRateB_ : TextureRateBPlug = PlugDescriptor("textureRateB")
	txrb_ : TextureRateBPlug = PlugDescriptor("textureRateB")
	textureRateG_ : TextureRateGPlug = PlugDescriptor("textureRateG")
	txrg_ : TextureRateGPlug = PlugDescriptor("textureRateG")
	textureRateR_ : TextureRateRPlug = PlugDescriptor("textureRateR")
	txrr_ : TextureRateRPlug = PlugDescriptor("textureRateR")
	node : PointEmitter = None
	pass
class UseLuminancePlug(Plug):
	node : PointEmitter = None
	pass
class UseRatePPPlug(Plug):
	node : PointEmitter = None
	pass
class VolumeEfficiencyPlug(Plug):
	node : PointEmitter = None
	pass
class VolumeOffsetXPlug(Plug):
	parent : VolumeOffsetPlug = PlugDescriptor("volumeOffset")
	node : PointEmitter = None
	pass
class VolumeOffsetYPlug(Plug):
	parent : VolumeOffsetPlug = PlugDescriptor("volumeOffset")
	node : PointEmitter = None
	pass
class VolumeOffsetZPlug(Plug):
	parent : VolumeOffsetPlug = PlugDescriptor("volumeOffset")
	node : PointEmitter = None
	pass
class VolumeOffsetPlug(Plug):
	volumeOffsetX_ : VolumeOffsetXPlug = PlugDescriptor("volumeOffsetX")
	vfx_ : VolumeOffsetXPlug = PlugDescriptor("volumeOffsetX")
	volumeOffsetY_ : VolumeOffsetYPlug = PlugDescriptor("volumeOffsetY")
	vfy_ : VolumeOffsetYPlug = PlugDescriptor("volumeOffsetY")
	volumeOffsetZ_ : VolumeOffsetZPlug = PlugDescriptor("volumeOffsetZ")
	vfz_ : VolumeOffsetZPlug = PlugDescriptor("volumeOffsetZ")
	node : PointEmitter = None
	pass
class VolumeShapePlug(Plug):
	node : PointEmitter = None
	pass
class VolumeSweepPlug(Plug):
	node : PointEmitter = None
	pass
# endregion


# define node class
class PointEmitter(DynBase):
	alongAxis_ : AlongAxisPlug = PlugDescriptor("alongAxis")
	aroundAxis_ : AroundAxisPlug = PlugDescriptor("aroundAxis")
	awayFromAxis_ : AwayFromAxisPlug = PlugDescriptor("awayFromAxis")
	awayFromCenter_ : AwayFromCenterPlug = PlugDescriptor("awayFromCenter")
	currentTime_ : CurrentTimePlug = PlugDescriptor("currentTime")
	cycleEmission_ : CycleEmissionPlug = PlugDescriptor("cycleEmission")
	cycleInterval_ : CycleIntervalPlug = PlugDescriptor("cycleInterval")
	deltaTime_ : DeltaTimePlug = PlugDescriptor("deltaTime")
	deltaTimeCycle_ : DeltaTimeCyclePlug = PlugDescriptor("deltaTimeCycle")
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	direction_ : DirectionPlug = PlugDescriptor("direction")
	directionalSpeed_ : DirectionalSpeedPlug = PlugDescriptor("directionalSpeed")
	displaySpeed_ : DisplaySpeedPlug = PlugDescriptor("displaySpeed")
	emitCountRemainder_ : EmitCountRemainderPlug = PlugDescriptor("emitCountRemainder")
	emitFromDark_ : EmitFromDarkPlug = PlugDescriptor("emitFromDark")
	emitterType_ : EmitterTypePlug = PlugDescriptor("emitterType")
	enableTextureRate_ : EnableTextureRatePlug = PlugDescriptor("enableTextureRate")
	inheritColor_ : InheritColorPlug = PlugDescriptor("inheritColor")
	inheritFactor_ : InheritFactorPlug = PlugDescriptor("inheritFactor")
	inheritOpacity_ : InheritOpacityPlug = PlugDescriptor("inheritOpacity")
	invertOpacity_ : InvertOpacityPlug = PlugDescriptor("invertOpacity")
	isFull_ : IsFullPlug = PlugDescriptor("isFull")
	maxDistance_ : MaxDistancePlug = PlugDescriptor("maxDistance")
	minDistance_ : MinDistancePlug = PlugDescriptor("minDistance")
	needParentUV_ : NeedParentUVPlug = PlugDescriptor("needParentUV")
	normalSpeed_ : NormalSpeedPlug = PlugDescriptor("normalSpeed")
	output_ : OutputPlug = PlugDescriptor("output")
	parentId_ : ParentIdPlug = PlugDescriptor("parentId")
	particleColorB_ : ParticleColorBPlug = PlugDescriptor("particleColorB")
	particleColorG_ : ParticleColorGPlug = PlugDescriptor("particleColorG")
	particleColorR_ : ParticleColorRPlug = PlugDescriptor("particleColorR")
	particleColor_ : ParticleColorPlug = PlugDescriptor("particleColor")
	randStateX_ : RandStateXPlug = PlugDescriptor("randStateX")
	randStateY_ : RandStateYPlug = PlugDescriptor("randStateY")
	randStateZ_ : RandStateZPlug = PlugDescriptor("randStateZ")
	randState_ : RandStatePlug = PlugDescriptor("randState")
	randomDirection_ : RandomDirectionPlug = PlugDescriptor("randomDirection")
	rate_ : RatePlug = PlugDescriptor("rate")
	ratePP_ : RatePPPlug = PlugDescriptor("ratePP")
	scaleRateByObjectSize_ : ScaleRateByObjectSizePlug = PlugDescriptor("scaleRateByObjectSize")
	scaleRateBySpeed_ : ScaleRateBySpeedPlug = PlugDescriptor("scaleRateBySpeed")
	scaleSpeedBySize_ : ScaleSpeedBySizePlug = PlugDescriptor("scaleSpeedBySize")
	sectionRadius_ : SectionRadiusPlug = PlugDescriptor("sectionRadius")
	seed_ : SeedPlug = PlugDescriptor("seed")
	speed_ : SpeedPlug = PlugDescriptor("speed")
	speedRandom_ : SpeedRandomPlug = PlugDescriptor("speedRandom")
	spread_ : SpreadPlug = PlugDescriptor("spread")
	startTime_ : StartTimePlug = PlugDescriptor("startTime")
	sweptGeometry_ : SweptGeometryPlug = PlugDescriptor("sweptGeometry")
	tangentSpeed_ : TangentSpeedPlug = PlugDescriptor("tangentSpeed")
	textureRateB_ : TextureRateBPlug = PlugDescriptor("textureRateB")
	textureRateG_ : TextureRateGPlug = PlugDescriptor("textureRateG")
	textureRateR_ : TextureRateRPlug = PlugDescriptor("textureRateR")
	textureRate_ : TextureRatePlug = PlugDescriptor("textureRate")
	useLuminance_ : UseLuminancePlug = PlugDescriptor("useLuminance")
	useRatePP_ : UseRatePPPlug = PlugDescriptor("useRatePP")
	volumeEfficiency_ : VolumeEfficiencyPlug = PlugDescriptor("volumeEfficiency")
	volumeOffsetX_ : VolumeOffsetXPlug = PlugDescriptor("volumeOffsetX")
	volumeOffsetY_ : VolumeOffsetYPlug = PlugDescriptor("volumeOffsetY")
	volumeOffsetZ_ : VolumeOffsetZPlug = PlugDescriptor("volumeOffsetZ")
	volumeOffset_ : VolumeOffsetPlug = PlugDescriptor("volumeOffset")
	volumeShape_ : VolumeShapePlug = PlugDescriptor("volumeShape")
	volumeSweep_ : VolumeSweepPlug = PlugDescriptor("volumeSweep")

	# node attributes

	typeName = "pointEmitter"
	typeIdInt = 1497713993
	nodeLeafClassAttrs = ["alongAxis", "aroundAxis", "awayFromAxis", "awayFromCenter", "currentTime", "cycleEmission", "cycleInterval", "deltaTime", "deltaTimeCycle", "directionX", "directionY", "directionZ", "direction", "directionalSpeed", "displaySpeed", "emitCountRemainder", "emitFromDark", "emitterType", "enableTextureRate", "inheritColor", "inheritFactor", "inheritOpacity", "invertOpacity", "isFull", "maxDistance", "minDistance", "needParentUV", "normalSpeed", "output", "parentId", "particleColorB", "particleColorG", "particleColorR", "particleColor", "randStateX", "randStateY", "randStateZ", "randState", "randomDirection", "rate", "ratePP", "scaleRateByObjectSize", "scaleRateBySpeed", "scaleSpeedBySize", "sectionRadius", "seed", "speed", "speedRandom", "spread", "startTime", "sweptGeometry", "tangentSpeed", "textureRateB", "textureRateG", "textureRateR", "textureRate", "useLuminance", "useRatePP", "volumeEfficiency", "volumeOffsetX", "volumeOffsetY", "volumeOffsetZ", "volumeOffset", "volumeShape", "volumeSweep"]
	nodeLeafPlugs = ["alongAxis", "aroundAxis", "awayFromAxis", "awayFromCenter", "currentTime", "cycleEmission", "cycleInterval", "deltaTime", "deltaTimeCycle", "direction", "directionalSpeed", "displaySpeed", "emitCountRemainder", "emitFromDark", "emitterType", "enableTextureRate", "inheritColor", "inheritFactor", "inheritOpacity", "invertOpacity", "isFull", "maxDistance", "minDistance", "needParentUV", "normalSpeed", "output", "parentId", "particleColor", "randState", "randomDirection", "rate", "ratePP", "scaleRateByObjectSize", "scaleRateBySpeed", "scaleSpeedBySize", "sectionRadius", "seed", "speed", "speedRandom", "spread", "startTime", "sweptGeometry", "tangentSpeed", "textureRate", "useLuminance", "useRatePP", "volumeEfficiency", "volumeOffset", "volumeShape", "volumeSweep"]
	pass

