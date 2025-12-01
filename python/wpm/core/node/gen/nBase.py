

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Particle = retriever.getNodeCls("Particle")
assert Particle
if T.TYPE_CHECKING:
	from .. import Particle

# add node doc



# region plug type defs
class ActivePlug(Plug):
	node : NBase = None
	pass
class AirPushDistancePlug(Plug):
	node : NBase = None
	pass
class AirPushVorticityPlug(Plug):
	node : NBase = None
	pass
class BouncePlug(Plug):
	node : NBase = None
	pass
class BounceMapPlug(Plug):
	node : NBase = None
	pass
class BounceMapTypePlug(Plug):
	node : NBase = None
	pass
class BouncePerVertexPlug(Plug):
	node : NBase = None
	pass
class CacheArrayDataPlug(Plug):
	node : NBase = None
	pass
class CollidePlug(Plug):
	node : NBase = None
	pass
class CollideStrengthPlug(Plug):
	node : NBase = None
	pass
class CollideStrengthMapPlug(Plug):
	node : NBase = None
	pass
class CollideStrengthMapTypePlug(Plug):
	node : NBase = None
	pass
class CollideStrengthPerVertexPlug(Plug):
	node : NBase = None
	pass
class CollisionFlagPlug(Plug):
	node : NBase = None
	pass
class CollisionLayerPlug(Plug):
	node : NBase = None
	pass
class CrossoverPushPlug(Plug):
	node : NBase = None
	pass
class CurrentStatePlug(Plug):
	node : NBase = None
	pass
class DampPlug(Plug):
	node : NBase = None
	pass
class DampMapPlug(Plug):
	node : NBase = None
	pass
class DampMapTypePlug(Plug):
	node : NBase = None
	pass
class DampPerVertexPlug(Plug):
	node : NBase = None
	pass
class DisplayColorBPlug(Plug):
	parent : DisplayColorPlug = PlugDescriptor("displayColor")
	node : NBase = None
	pass
class DisplayColorGPlug(Plug):
	parent : DisplayColorPlug = PlugDescriptor("displayColor")
	node : NBase = None
	pass
class DisplayColorRPlug(Plug):
	parent : DisplayColorPlug = PlugDescriptor("displayColor")
	node : NBase = None
	pass
class DisplayColorPlug(Plug):
	displayColorB_ : DisplayColorBPlug = PlugDescriptor("displayColorB")
	dcb_ : DisplayColorBPlug = PlugDescriptor("displayColorB")
	displayColorG_ : DisplayColorGPlug = PlugDescriptor("displayColorG")
	dcg_ : DisplayColorGPlug = PlugDescriptor("displayColorG")
	displayColorR_ : DisplayColorRPlug = PlugDescriptor("displayColorR")
	dcr_ : DisplayColorRPlug = PlugDescriptor("displayColorR")
	node : NBase = None
	pass
class FieldDistancePlug(Plug):
	node : NBase = None
	pass
class FieldMagnitudePlug(Plug):
	node : NBase = None
	pass
class FieldMagnitudeMapPlug(Plug):
	node : NBase = None
	pass
class FieldMagnitudeMapTypePlug(Plug):
	node : NBase = None
	pass
class FieldMagnitudePerVertexPlug(Plug):
	node : NBase = None
	pass
class FieldScale_FloatValuePlug(Plug):
	parent : FieldScalePlug = PlugDescriptor("fieldScale")
	node : NBase = None
	pass
class FieldScale_InterpPlug(Plug):
	parent : FieldScalePlug = PlugDescriptor("fieldScale")
	node : NBase = None
	pass
class FieldScale_PositionPlug(Plug):
	parent : FieldScalePlug = PlugDescriptor("fieldScale")
	node : NBase = None
	pass
class FieldScalePlug(Plug):
	fieldScale_FloatValue_ : FieldScale_FloatValuePlug = PlugDescriptor("fieldScale_FloatValue")
	fscfv_ : FieldScale_FloatValuePlug = PlugDescriptor("fieldScale_FloatValue")
	fieldScale_Interp_ : FieldScale_InterpPlug = PlugDescriptor("fieldScale_Interp")
	fsci_ : FieldScale_InterpPlug = PlugDescriptor("fieldScale_Interp")
	fieldScale_Position_ : FieldScale_PositionPlug = PlugDescriptor("fieldScale_Position")
	fscp_ : FieldScale_PositionPlug = PlugDescriptor("fieldScale_Position")
	node : NBase = None
	pass
class ForceFieldPlug(Plug):
	node : NBase = None
	pass
class FrictionPlug(Plug):
	node : NBase = None
	pass
class FrictionMapPlug(Plug):
	node : NBase = None
	pass
class FrictionMapTypePlug(Plug):
	node : NBase = None
	pass
class FrictionPerVertexPlug(Plug):
	node : NBase = None
	pass
class InputMeshPlug(Plug):
	node : NBase = None
	pass
class InternalStatePlug(Plug):
	node : NBase = None
	pass
class LastNBaseTimePlug(Plug):
	node : NBase = None
	pass
class LocalForceXPlug(Plug):
	parent : LocalForcePlug = PlugDescriptor("localForce")
	node : NBase = None
	pass
class LocalForceYPlug(Plug):
	parent : LocalForcePlug = PlugDescriptor("localForce")
	node : NBase = None
	pass
class LocalForceZPlug(Plug):
	parent : LocalForcePlug = PlugDescriptor("localForce")
	node : NBase = None
	pass
class LocalForcePlug(Plug):
	localForceX_ : LocalForceXPlug = PlugDescriptor("localForceX")
	lfcx_ : LocalForceXPlug = PlugDescriptor("localForceX")
	localForceY_ : LocalForceYPlug = PlugDescriptor("localForceY")
	lfcy_ : LocalForceYPlug = PlugDescriptor("localForceY")
	localForceZ_ : LocalForceZPlug = PlugDescriptor("localForceZ")
	lfcz_ : LocalForceZPlug = PlugDescriptor("localForceZ")
	node : NBase = None
	pass
class LocalSpaceOutputPlug(Plug):
	node : NBase = None
	pass
class LocalWindXPlug(Plug):
	parent : LocalWindPlug = PlugDescriptor("localWind")
	node : NBase = None
	pass
class LocalWindYPlug(Plug):
	parent : LocalWindPlug = PlugDescriptor("localWind")
	node : NBase = None
	pass
class LocalWindZPlug(Plug):
	parent : LocalWindPlug = PlugDescriptor("localWind")
	node : NBase = None
	pass
class LocalWindPlug(Plug):
	localWindX_ : LocalWindXPlug = PlugDescriptor("localWindX")
	lwnx_ : LocalWindXPlug = PlugDescriptor("localWindX")
	localWindY_ : LocalWindYPlug = PlugDescriptor("localWindY")
	lwny_ : LocalWindYPlug = PlugDescriptor("localWindY")
	localWindZ_ : LocalWindZPlug = PlugDescriptor("localWindZ")
	lwnz_ : LocalWindZPlug = PlugDescriptor("localWindZ")
	node : NBase = None
	pass
class MappedMassPlug(Plug):
	node : NBase = None
	pass
class MassMapPlug(Plug):
	node : NBase = None
	pass
class MassMapTypePlug(Plug):
	node : NBase = None
	pass
class MassPerVertexPlug(Plug):
	node : NBase = None
	pass
class MaxIterationsPlug(Plug):
	node : NBase = None
	pass
class MaxSelfCollisionIterationsPlug(Plug):
	node : NBase = None
	pass
class NextStatePlug(Plug):
	node : NBase = None
	pass
class NucleusIdPlug(Plug):
	node : NBase = None
	pass
class PlayFromCachePlug(Plug):
	node : NBase = None
	pass
class PointFieldDistancePlug(Plug):
	node : NBase = None
	pass
class PointFieldDropoff_FloatValuePlug(Plug):
	parent : PointFieldDropoffPlug = PlugDescriptor("pointFieldDropoff")
	node : NBase = None
	pass
class PointFieldDropoff_InterpPlug(Plug):
	parent : PointFieldDropoffPlug = PlugDescriptor("pointFieldDropoff")
	node : NBase = None
	pass
class PointFieldDropoff_PositionPlug(Plug):
	parent : PointFieldDropoffPlug = PlugDescriptor("pointFieldDropoff")
	node : NBase = None
	pass
class PointFieldDropoffPlug(Plug):
	pointFieldDropoff_FloatValue_ : PointFieldDropoff_FloatValuePlug = PlugDescriptor("pointFieldDropoff_FloatValue")
	pfdofv_ : PointFieldDropoff_FloatValuePlug = PlugDescriptor("pointFieldDropoff_FloatValue")
	pointFieldDropoff_Interp_ : PointFieldDropoff_InterpPlug = PlugDescriptor("pointFieldDropoff_Interp")
	pfdoi_ : PointFieldDropoff_InterpPlug = PlugDescriptor("pointFieldDropoff_Interp")
	pointFieldDropoff_Position_ : PointFieldDropoff_PositionPlug = PlugDescriptor("pointFieldDropoff_Position")
	pfdop_ : PointFieldDropoff_PositionPlug = PlugDescriptor("pointFieldDropoff_Position")
	node : NBase = None
	pass
class PointFieldMagnitudePlug(Plug):
	node : NBase = None
	pass
class PointForceFieldPlug(Plug):
	node : NBase = None
	pass
class PointMassPlug(Plug):
	node : NBase = None
	pass
class PositionsPlug(Plug):
	node : NBase = None
	pass
class PushOutPlug(Plug):
	node : NBase = None
	pass
class PushOutRadiusPlug(Plug):
	node : NBase = None
	pass
class RestLengthScalePlug(Plug):
	node : NBase = None
	pass
class SelfAttractPlug(Plug):
	node : NBase = None
	pass
class SelfCollidePlug(Plug):
	node : NBase = None
	pass
class SelfCollisionFlagPlug(Plug):
	node : NBase = None
	pass
class StartPositionsPlug(Plug):
	node : NBase = None
	pass
class StartStatePlug(Plug):
	node : NBase = None
	pass
class StartVelocitiesPlug(Plug):
	node : NBase = None
	pass
class StickinessPlug(Plug):
	node : NBase = None
	pass
class StickinessMapPlug(Plug):
	node : NBase = None
	pass
class StickinessMapTypePlug(Plug):
	node : NBase = None
	pass
class StickinessPerVertexPlug(Plug):
	node : NBase = None
	pass
class ThicknessPlug(Plug):
	node : NBase = None
	pass
class ThicknessMapPlug(Plug):
	node : NBase = None
	pass
class ThicknessMapTypePlug(Plug):
	node : NBase = None
	pass
class ThicknessPerVertexPlug(Plug):
	node : NBase = None
	pass
class TrappedCheckPlug(Plug):
	node : NBase = None
	pass
class VelocitiesPlug(Plug):
	node : NBase = None
	pass
class WindShadowDiffusionPlug(Plug):
	node : NBase = None
	pass
class WindShadowDistancePlug(Plug):
	node : NBase = None
	pass
# endregion


# define node class
class NBase(Particle):
	active_ : ActivePlug = PlugDescriptor("active")
	airPushDistance_ : AirPushDistancePlug = PlugDescriptor("airPushDistance")
	airPushVorticity_ : AirPushVorticityPlug = PlugDescriptor("airPushVorticity")
	bounce_ : BouncePlug = PlugDescriptor("bounce")
	bounceMap_ : BounceMapPlug = PlugDescriptor("bounceMap")
	bounceMapType_ : BounceMapTypePlug = PlugDescriptor("bounceMapType")
	bouncePerVertex_ : BouncePerVertexPlug = PlugDescriptor("bouncePerVertex")
	cacheArrayData_ : CacheArrayDataPlug = PlugDescriptor("cacheArrayData")
	collide_ : CollidePlug = PlugDescriptor("collide")
	collideStrength_ : CollideStrengthPlug = PlugDescriptor("collideStrength")
	collideStrengthMap_ : CollideStrengthMapPlug = PlugDescriptor("collideStrengthMap")
	collideStrengthMapType_ : CollideStrengthMapTypePlug = PlugDescriptor("collideStrengthMapType")
	collideStrengthPerVertex_ : CollideStrengthPerVertexPlug = PlugDescriptor("collideStrengthPerVertex")
	collisionFlag_ : CollisionFlagPlug = PlugDescriptor("collisionFlag")
	collisionLayer_ : CollisionLayerPlug = PlugDescriptor("collisionLayer")
	crossoverPush_ : CrossoverPushPlug = PlugDescriptor("crossoverPush")
	currentState_ : CurrentStatePlug = PlugDescriptor("currentState")
	damp_ : DampPlug = PlugDescriptor("damp")
	dampMap_ : DampMapPlug = PlugDescriptor("dampMap")
	dampMapType_ : DampMapTypePlug = PlugDescriptor("dampMapType")
	dampPerVertex_ : DampPerVertexPlug = PlugDescriptor("dampPerVertex")
	displayColorB_ : DisplayColorBPlug = PlugDescriptor("displayColorB")
	displayColorG_ : DisplayColorGPlug = PlugDescriptor("displayColorG")
	displayColorR_ : DisplayColorRPlug = PlugDescriptor("displayColorR")
	displayColor_ : DisplayColorPlug = PlugDescriptor("displayColor")
	fieldDistance_ : FieldDistancePlug = PlugDescriptor("fieldDistance")
	fieldMagnitude_ : FieldMagnitudePlug = PlugDescriptor("fieldMagnitude")
	fieldMagnitudeMap_ : FieldMagnitudeMapPlug = PlugDescriptor("fieldMagnitudeMap")
	fieldMagnitudeMapType_ : FieldMagnitudeMapTypePlug = PlugDescriptor("fieldMagnitudeMapType")
	fieldMagnitudePerVertex_ : FieldMagnitudePerVertexPlug = PlugDescriptor("fieldMagnitudePerVertex")
	fieldScale_FloatValue_ : FieldScale_FloatValuePlug = PlugDescriptor("fieldScale_FloatValue")
	fieldScale_Interp_ : FieldScale_InterpPlug = PlugDescriptor("fieldScale_Interp")
	fieldScale_Position_ : FieldScale_PositionPlug = PlugDescriptor("fieldScale_Position")
	fieldScale_ : FieldScalePlug = PlugDescriptor("fieldScale")
	forceField_ : ForceFieldPlug = PlugDescriptor("forceField")
	friction_ : FrictionPlug = PlugDescriptor("friction")
	frictionMap_ : FrictionMapPlug = PlugDescriptor("frictionMap")
	frictionMapType_ : FrictionMapTypePlug = PlugDescriptor("frictionMapType")
	frictionPerVertex_ : FrictionPerVertexPlug = PlugDescriptor("frictionPerVertex")
	inputMesh_ : InputMeshPlug = PlugDescriptor("inputMesh")
	internalState_ : InternalStatePlug = PlugDescriptor("internalState")
	lastNBaseTime_ : LastNBaseTimePlug = PlugDescriptor("lastNBaseTime")
	localForceX_ : LocalForceXPlug = PlugDescriptor("localForceX")
	localForceY_ : LocalForceYPlug = PlugDescriptor("localForceY")
	localForceZ_ : LocalForceZPlug = PlugDescriptor("localForceZ")
	localForce_ : LocalForcePlug = PlugDescriptor("localForce")
	localSpaceOutput_ : LocalSpaceOutputPlug = PlugDescriptor("localSpaceOutput")
	localWindX_ : LocalWindXPlug = PlugDescriptor("localWindX")
	localWindY_ : LocalWindYPlug = PlugDescriptor("localWindY")
	localWindZ_ : LocalWindZPlug = PlugDescriptor("localWindZ")
	localWind_ : LocalWindPlug = PlugDescriptor("localWind")
	mappedMass_ : MappedMassPlug = PlugDescriptor("mappedMass")
	massMap_ : MassMapPlug = PlugDescriptor("massMap")
	massMapType_ : MassMapTypePlug = PlugDescriptor("massMapType")
	massPerVertex_ : MassPerVertexPlug = PlugDescriptor("massPerVertex")
	maxIterations_ : MaxIterationsPlug = PlugDescriptor("maxIterations")
	maxSelfCollisionIterations_ : MaxSelfCollisionIterationsPlug = PlugDescriptor("maxSelfCollisionIterations")
	nextState_ : NextStatePlug = PlugDescriptor("nextState")
	nucleusId_ : NucleusIdPlug = PlugDescriptor("nucleusId")
	playFromCache_ : PlayFromCachePlug = PlugDescriptor("playFromCache")
	pointFieldDistance_ : PointFieldDistancePlug = PlugDescriptor("pointFieldDistance")
	pointFieldDropoff_FloatValue_ : PointFieldDropoff_FloatValuePlug = PlugDescriptor("pointFieldDropoff_FloatValue")
	pointFieldDropoff_Interp_ : PointFieldDropoff_InterpPlug = PlugDescriptor("pointFieldDropoff_Interp")
	pointFieldDropoff_Position_ : PointFieldDropoff_PositionPlug = PlugDescriptor("pointFieldDropoff_Position")
	pointFieldDropoff_ : PointFieldDropoffPlug = PlugDescriptor("pointFieldDropoff")
	pointFieldMagnitude_ : PointFieldMagnitudePlug = PlugDescriptor("pointFieldMagnitude")
	pointForceField_ : PointForceFieldPlug = PlugDescriptor("pointForceField")
	pointMass_ : PointMassPlug = PlugDescriptor("pointMass")
	positions_ : PositionsPlug = PlugDescriptor("positions")
	pushOut_ : PushOutPlug = PlugDescriptor("pushOut")
	pushOutRadius_ : PushOutRadiusPlug = PlugDescriptor("pushOutRadius")
	restLengthScale_ : RestLengthScalePlug = PlugDescriptor("restLengthScale")
	selfAttract_ : SelfAttractPlug = PlugDescriptor("selfAttract")
	selfCollide_ : SelfCollidePlug = PlugDescriptor("selfCollide")
	selfCollisionFlag_ : SelfCollisionFlagPlug = PlugDescriptor("selfCollisionFlag")
	startPositions_ : StartPositionsPlug = PlugDescriptor("startPositions")
	startState_ : StartStatePlug = PlugDescriptor("startState")
	startVelocities_ : StartVelocitiesPlug = PlugDescriptor("startVelocities")
	stickiness_ : StickinessPlug = PlugDescriptor("stickiness")
	stickinessMap_ : StickinessMapPlug = PlugDescriptor("stickinessMap")
	stickinessMapType_ : StickinessMapTypePlug = PlugDescriptor("stickinessMapType")
	stickinessPerVertex_ : StickinessPerVertexPlug = PlugDescriptor("stickinessPerVertex")
	thickness_ : ThicknessPlug = PlugDescriptor("thickness")
	thicknessMap_ : ThicknessMapPlug = PlugDescriptor("thicknessMap")
	thicknessMapType_ : ThicknessMapTypePlug = PlugDescriptor("thicknessMapType")
	thicknessPerVertex_ : ThicknessPerVertexPlug = PlugDescriptor("thicknessPerVertex")
	trappedCheck_ : TrappedCheckPlug = PlugDescriptor("trappedCheck")
	velocities_ : VelocitiesPlug = PlugDescriptor("velocities")
	windShadowDiffusion_ : WindShadowDiffusionPlug = PlugDescriptor("windShadowDiffusion")
	windShadowDistance_ : WindShadowDistancePlug = PlugDescriptor("windShadowDistance")

	# node attributes

	typeName = "nBase"
	apiTypeInt = 998
	apiTypeStr = "kNBase"
	typeIdInt = 1312964947
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["active", "airPushDistance", "airPushVorticity", "bounce", "bounceMap", "bounceMapType", "bouncePerVertex", "cacheArrayData", "collide", "collideStrength", "collideStrengthMap", "collideStrengthMapType", "collideStrengthPerVertex", "collisionFlag", "collisionLayer", "crossoverPush", "currentState", "damp", "dampMap", "dampMapType", "dampPerVertex", "displayColorB", "displayColorG", "displayColorR", "displayColor", "fieldDistance", "fieldMagnitude", "fieldMagnitudeMap", "fieldMagnitudeMapType", "fieldMagnitudePerVertex", "fieldScale_FloatValue", "fieldScale_Interp", "fieldScale_Position", "fieldScale", "forceField", "friction", "frictionMap", "frictionMapType", "frictionPerVertex", "inputMesh", "internalState", "lastNBaseTime", "localForceX", "localForceY", "localForceZ", "localForce", "localSpaceOutput", "localWindX", "localWindY", "localWindZ", "localWind", "mappedMass", "massMap", "massMapType", "massPerVertex", "maxIterations", "maxSelfCollisionIterations", "nextState", "nucleusId", "playFromCache", "pointFieldDistance", "pointFieldDropoff_FloatValue", "pointFieldDropoff_Interp", "pointFieldDropoff_Position", "pointFieldDropoff", "pointFieldMagnitude", "pointForceField", "pointMass", "positions", "pushOut", "pushOutRadius", "restLengthScale", "selfAttract", "selfCollide", "selfCollisionFlag", "startPositions", "startState", "startVelocities", "stickiness", "stickinessMap", "stickinessMapType", "stickinessPerVertex", "thickness", "thicknessMap", "thicknessMapType", "thicknessPerVertex", "trappedCheck", "velocities", "windShadowDiffusion", "windShadowDistance"]
	nodeLeafPlugs = ["active", "airPushDistance", "airPushVorticity", "bounce", "bounceMap", "bounceMapType", "bouncePerVertex", "cacheArrayData", "collide", "collideStrength", "collideStrengthMap", "collideStrengthMapType", "collideStrengthPerVertex", "collisionFlag", "collisionLayer", "crossoverPush", "currentState", "damp", "dampMap", "dampMapType", "dampPerVertex", "displayColor", "fieldDistance", "fieldMagnitude", "fieldMagnitudeMap", "fieldMagnitudeMapType", "fieldMagnitudePerVertex", "fieldScale", "forceField", "friction", "frictionMap", "frictionMapType", "frictionPerVertex", "inputMesh", "internalState", "lastNBaseTime", "localForce", "localSpaceOutput", "localWind", "mappedMass", "massMap", "massMapType", "massPerVertex", "maxIterations", "maxSelfCollisionIterations", "nextState", "nucleusId", "playFromCache", "pointFieldDistance", "pointFieldDropoff", "pointFieldMagnitude", "pointForceField", "pointMass", "positions", "pushOut", "pushOutRadius", "restLengthScale", "selfAttract", "selfCollide", "selfCollisionFlag", "startPositions", "startState", "startVelocities", "stickiness", "stickinessMap", "stickinessMapType", "stickinessPerVertex", "thickness", "thicknessMap", "thicknessMapType", "thicknessPerVertex", "trappedCheck", "velocities", "windShadowDiffusion", "windShadowDistance"]
	pass

