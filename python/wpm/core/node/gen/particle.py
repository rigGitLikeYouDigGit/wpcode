

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
DeformableShape = retriever.getNodeCls("DeformableShape")
assert DeformableShape
if T.TYPE_CHECKING:
	from .. import DeformableShape

# add node doc



# region plug type defs
class AccelerationPlug(Plug):
	node : Particle = None
	pass
class Acceleration0Plug(Plug):
	node : Particle = None
	pass
class AgePlug(Plug):
	node : Particle = None
	pass
class Age0Plug(Plug):
	node : Particle = None
	pass
class AgeCachePlug(Plug):
	node : Particle = None
	pass
class AgesLastDonePlug(Plug):
	node : Particle = None
	pass
class AuxiliariesOwnedPlug(Plug):
	node : Particle = None
	pass
class BirthTimePlug(Plug):
	node : Particle = None
	pass
class BirthTime0Plug(Plug):
	node : Particle = None
	pass
class BirthTimeCachePlug(Plug):
	node : Particle = None
	pass
class CacheDataPlug(Plug):
	node : Particle = None
	pass
class CacheWidthPlug(Plug):
	node : Particle = None
	pass
class CachedPositionPlug(Plug):
	node : Particle = None
	pass
class CachedTimePlug(Plug):
	node : Particle = None
	pass
class CachedVelocityPlug(Plug):
	node : Particle = None
	pass
class CachedWorldCentroidXPlug(Plug):
	parent : CachedWorldCentroidPlug = PlugDescriptor("cachedWorldCentroid")
	node : Particle = None
	pass
class CachedWorldCentroidYPlug(Plug):
	parent : CachedWorldCentroidPlug = PlugDescriptor("cachedWorldCentroid")
	node : Particle = None
	pass
class CachedWorldCentroidZPlug(Plug):
	parent : CachedWorldCentroidPlug = PlugDescriptor("cachedWorldCentroid")
	node : Particle = None
	pass
class CachedWorldCentroidPlug(Plug):
	cachedWorldCentroidX_ : CachedWorldCentroidXPlug = PlugDescriptor("cachedWorldCentroidX")
	cwcx_ : CachedWorldCentroidXPlug = PlugDescriptor("cachedWorldCentroidX")
	cachedWorldCentroidY_ : CachedWorldCentroidYPlug = PlugDescriptor("cachedWorldCentroidY")
	cwcy_ : CachedWorldCentroidYPlug = PlugDescriptor("cachedWorldCentroidY")
	cachedWorldCentroidZ_ : CachedWorldCentroidZPlug = PlugDescriptor("cachedWorldCentroidZ")
	cwcz_ : CachedWorldCentroidZPlug = PlugDescriptor("cachedWorldCentroidZ")
	node : Particle = None
	pass
class CachedWorldPositionPlug(Plug):
	node : Particle = None
	pass
class CachedWorldVelocityPlug(Plug):
	node : Particle = None
	pass
class CentroidXPlug(Plug):
	parent : CentroidPlug = PlugDescriptor("centroid")
	node : Particle = None
	pass
class CentroidYPlug(Plug):
	parent : CentroidPlug = PlugDescriptor("centroid")
	node : Particle = None
	pass
class CentroidZPlug(Plug):
	parent : CentroidPlug = PlugDescriptor("centroid")
	node : Particle = None
	pass
class CentroidPlug(Plug):
	centroidX_ : CentroidXPlug = PlugDescriptor("centroidX")
	ctdx_ : CentroidXPlug = PlugDescriptor("centroidX")
	centroidY_ : CentroidYPlug = PlugDescriptor("centroidY")
	ctdy_ : CentroidYPlug = PlugDescriptor("centroidY")
	centroidZ_ : CentroidZPlug = PlugDescriptor("centroidZ")
	ctdz_ : CentroidZPlug = PlugDescriptor("centroidZ")
	node : Particle = None
	pass
class CollisionConnectionsPlug(Plug):
	node : Particle = None
	pass
class CollisionFrictionPlug(Plug):
	parent : CollisionDataPlug = PlugDescriptor("collisionData")
	node : Particle = None
	pass
class CollisionGeometryPlug(Plug):
	parent : CollisionDataPlug = PlugDescriptor("collisionData")
	node : Particle = None
	pass
class CollisionOffsetPlug(Plug):
	parent : CollisionDataPlug = PlugDescriptor("collisionData")
	node : Particle = None
	pass
class CollisionResiliencePlug(Plug):
	parent : CollisionDataPlug = PlugDescriptor("collisionData")
	node : Particle = None
	pass
class CollisionDataPlug(Plug):
	collisionFriction_ : CollisionFrictionPlug = PlugDescriptor("collisionFriction")
	cfr_ : CollisionFrictionPlug = PlugDescriptor("collisionFriction")
	collisionGeometry_ : CollisionGeometryPlug = PlugDescriptor("collisionGeometry")
	cge_ : CollisionGeometryPlug = PlugDescriptor("collisionGeometry")
	collisionOffset_ : CollisionOffsetPlug = PlugDescriptor("collisionOffset")
	cof_ : CollisionOffsetPlug = PlugDescriptor("collisionOffset")
	collisionResilience_ : CollisionResiliencePlug = PlugDescriptor("collisionResilience")
	crs_ : CollisionResiliencePlug = PlugDescriptor("collisionResilience")
	node : Particle = None
	pass
class CollisionEventsPlug(Plug):
	node : Particle = None
	pass
class CollisionRecordsPlug(Plug):
	node : Particle = None
	pass
class CollisionsPlug(Plug):
	node : Particle = None
	pass
class ComputingCountPlug(Plug):
	node : Particle = None
	pass
class ConnectionsToMePlug(Plug):
	node : Particle = None
	pass
class ConservePlug(Plug):
	node : Particle = None
	pass
class CountPlug(Plug):
	node : Particle = None
	pass
class CurrentParticlePlug(Plug):
	node : Particle = None
	pass
class CurrentSceneTimePlug(Plug):
	node : Particle = None
	pass
class CurrentTimePlug(Plug):
	node : Particle = None
	pass
class CurrentTimeSavePlug(Plug):
	node : Particle = None
	pass
class DeathPlug(Plug):
	node : Particle = None
	pass
class DebugDrawPlug(Plug):
	node : Particle = None
	pass
class DeformedPositionPlug(Plug):
	node : Particle = None
	pass
class DepthSortPlug(Plug):
	node : Particle = None
	pass
class DieOnEmissionVolumeExitPlug(Plug):
	node : Particle = None
	pass
class DiedLastTimePlug(Plug):
	node : Particle = None
	pass
class DisableCloudAxisPlug(Plug):
	node : Particle = None
	pass
class DoAgePlug(Plug):
	node : Particle = None
	pass
class DoDynamicsPlug(Plug):
	node : Particle = None
	pass
class DoEmissionPlug(Plug):
	node : Particle = None
	pass
class DynamicsWeightPlug(Plug):
	node : Particle = None
	pass
class EmissionPlug(Plug):
	node : Particle = None
	pass
class EmissionInWorldPlug(Plug):
	node : Particle = None
	pass
class EmitterConnectionsPlug(Plug):
	node : Particle = None
	pass
class EmitterDataDeltaTimePlug(Plug):
	parent : EmitterDataPlug = PlugDescriptor("emitterData")
	node : Particle = None
	pass
class EmitterDataPositionPlug(Plug):
	parent : EmitterDataPlug = PlugDescriptor("emitterData")
	node : Particle = None
	pass
class EmitterDataVelocityPlug(Plug):
	parent : EmitterDataPlug = PlugDescriptor("emitterData")
	node : Particle = None
	pass
class EmitterDataPlug(Plug):
	emitterDataDeltaTime_ : EmitterDataDeltaTimePlug = PlugDescriptor("emitterDataDeltaTime")
	edt_ : EmitterDataDeltaTimePlug = PlugDescriptor("emitterDataDeltaTime")
	emitterDataPosition_ : EmitterDataPositionPlug = PlugDescriptor("emitterDataPosition")
	edp_ : EmitterDataPositionPlug = PlugDescriptor("emitterDataPosition")
	emitterDataVelocity_ : EmitterDataVelocityPlug = PlugDescriptor("emitterDataVelocity")
	edv_ : EmitterDataVelocityPlug = PlugDescriptor("emitterDataVelocity")
	node : Particle = None
	pass
class EmitterIdPlug(Plug):
	node : Particle = None
	pass
class EmitterId0Plug(Plug):
	node : Particle = None
	pass
class EnforceCountFromHistoryPlug(Plug):
	node : Particle = None
	pass
class EvaluationTimePlug(Plug):
	node : Particle = None
	pass
class EventCountPlug(Plug):
	node : Particle = None
	pass
class EventDiePlug(Plug):
	node : Particle = None
	pass
class EventEmitPlug(Plug):
	node : Particle = None
	pass
class EventNamePlug(Plug):
	node : Particle = None
	pass
class EventNameCountPlug(Plug):
	node : Particle = None
	pass
class EventProcPlug(Plug):
	node : Particle = None
	pass
class EventRandStateXPlug(Plug):
	parent : EventRandStatePlug = PlugDescriptor("eventRandState")
	node : Particle = None
	pass
class EventRandStateYPlug(Plug):
	parent : EventRandStatePlug = PlugDescriptor("eventRandState")
	node : Particle = None
	pass
class EventRandStateZPlug(Plug):
	parent : EventRandStatePlug = PlugDescriptor("eventRandState")
	node : Particle = None
	pass
class EventRandStatePlug(Plug):
	eventRandStateX_ : EventRandStateXPlug = PlugDescriptor("eventRandStateX")
	ersx_ : EventRandStateXPlug = PlugDescriptor("eventRandStateX")
	eventRandStateY_ : EventRandStateYPlug = PlugDescriptor("eventRandStateY")
	ersy_ : EventRandStateYPlug = PlugDescriptor("eventRandStateY")
	eventRandStateZ_ : EventRandStateZPlug = PlugDescriptor("eventRandStateZ")
	ersz_ : EventRandStateZPlug = PlugDescriptor("eventRandStateZ")
	node : Particle = None
	pass
class EventRandomPlug(Plug):
	node : Particle = None
	pass
class EventSeedPlug(Plug):
	node : Particle = None
	pass
class EventSplitPlug(Plug):
	node : Particle = None
	pass
class EventSpreadPlug(Plug):
	node : Particle = None
	pass
class EventTargetPlug(Plug):
	node : Particle = None
	pass
class EventTestPlug(Plug):
	node : Particle = None
	pass
class EventValidPlug(Plug):
	node : Particle = None
	pass
class ExecuteCreationExpressionPlug(Plug):
	node : Particle = None
	pass
class ExecuteRuntimeAfterDynamicsExpressionPlug(Plug):
	node : Particle = None
	pass
class ExecuteRuntimeBeforeDynamicsExpressionPlug(Plug):
	node : Particle = None
	pass
class ExpressionsAfterDynamicsPlug(Plug):
	node : Particle = None
	pass
class FieldConnectionsPlug(Plug):
	node : Particle = None
	pass
class FieldDataDeltaTimePlug(Plug):
	parent : FieldDataPlug = PlugDescriptor("fieldData")
	node : Particle = None
	pass
class FieldDataMassPlug(Plug):
	parent : FieldDataPlug = PlugDescriptor("fieldData")
	node : Particle = None
	pass
class FieldDataPositionPlug(Plug):
	parent : FieldDataPlug = PlugDescriptor("fieldData")
	node : Particle = None
	pass
class FieldDataVelocityPlug(Plug):
	parent : FieldDataPlug = PlugDescriptor("fieldData")
	node : Particle = None
	pass
class FieldDataPlug(Plug):
	fieldDataDeltaTime_ : FieldDataDeltaTimePlug = PlugDescriptor("fieldDataDeltaTime")
	fdt_ : FieldDataDeltaTimePlug = PlugDescriptor("fieldDataDeltaTime")
	fieldDataMass_ : FieldDataMassPlug = PlugDescriptor("fieldDataMass")
	fdm_ : FieldDataMassPlug = PlugDescriptor("fieldDataMass")
	fieldDataPosition_ : FieldDataPositionPlug = PlugDescriptor("fieldDataPosition")
	fdp_ : FieldDataPositionPlug = PlugDescriptor("fieldDataPosition")
	fieldDataVelocity_ : FieldDataVelocityPlug = PlugDescriptor("fieldDataVelocity")
	fdv_ : FieldDataVelocityPlug = PlugDescriptor("fieldDataVelocity")
	node : Particle = None
	pass
class FinalLifespanPPPlug(Plug):
	node : Particle = None
	pass
class ForcePlug(Plug):
	node : Particle = None
	pass
class ForceDynamicsPlug(Plug):
	node : Particle = None
	pass
class ForceEmissionPlug(Plug):
	node : Particle = None
	pass
class ForcesInWorldPlug(Plug):
	node : Particle = None
	pass
class FramePlug(Plug):
	node : Particle = None
	pass
class GeneralSeedPlug(Plug):
	node : Particle = None
	pass
class GoalActivePlug(Plug):
	node : Particle = None
	pass
class GoalGeometryPlug(Plug):
	node : Particle = None
	pass
class GoalSmoothnessPlug(Plug):
	node : Particle = None
	pass
class GoalUvSetNamePlug(Plug):
	node : Particle = None
	pass
class GoalWeightPlug(Plug):
	node : Particle = None
	pass
class IdCachePlug(Plug):
	node : Particle = None
	pass
class IdIndexPlug(Plug):
	parent : IdMappingPlug = PlugDescriptor("idMapping")
	node : Particle = None
	pass
class SortedIdPlug(Plug):
	parent : IdMappingPlug = PlugDescriptor("idMapping")
	node : Particle = None
	pass
class IdMappingPlug(Plug):
	idIndex_ : IdIndexPlug = PlugDescriptor("idIndex")
	idix_ : IdIndexPlug = PlugDescriptor("idIndex")
	sortedId_ : SortedIdPlug = PlugDescriptor("sortedId")
	sid_ : SortedIdPlug = PlugDescriptor("sortedId")
	node : Particle = None
	pass
class InheritColorPlug(Plug):
	node : Particle = None
	pass
class InheritFactorPlug(Plug):
	node : Particle = None
	pass
class InputPlug(Plug):
	node : Particle = None
	pass
class InputForcePlug(Plug):
	node : Particle = None
	pass
class InputGeometryPlug(Plug):
	node : Particle = None
	pass
class InputGeometryPointsPlug(Plug):
	node : Particle = None
	pass
class InputGeometrySpacePlug(Plug):
	node : Particle = None
	pass
class InstanceAttributeMappingPlug(Plug):
	parent : InstanceDataPlug = PlugDescriptor("instanceData")
	node : Particle = None
	pass
class InstancePointDataPlug(Plug):
	parent : InstanceDataPlug = PlugDescriptor("instanceData")
	node : Particle = None
	pass
class InstanceDataPlug(Plug):
	instanceAttributeMapping_ : InstanceAttributeMappingPlug = PlugDescriptor("instanceAttributeMapping")
	iam_ : InstanceAttributeMappingPlug = PlugDescriptor("instanceAttributeMapping")
	instancePointData_ : InstancePointDataPlug = PlugDescriptor("instancePointData")
	ipd_ : InstancePointDataPlug = PlugDescriptor("instancePointData")
	node : Particle = None
	pass
class InternalCreationExpressionPlug(Plug):
	node : Particle = None
	pass
class InternalRuntimeAfterDynamicsExpressionPlug(Plug):
	node : Particle = None
	pass
class InternalRuntimeBeforeDynamicsExpressionPlug(Plug):
	node : Particle = None
	pass
class InternalRuntimeExpressionPlug(Plug):
	node : Particle = None
	pass
class IsDynamicPlug(Plug):
	node : Particle = None
	pass
class IsFullPlug(Plug):
	node : Particle = None
	pass
class LastCachedPositionPlug(Plug):
	node : Particle = None
	pass
class LastPositionPlug(Plug):
	node : Particle = None
	pass
class LastSceneTimePlug(Plug):
	node : Particle = None
	pass
class LastTimeEvaluatedPlug(Plug):
	node : Particle = None
	pass
class LastTotalEventCountPlug(Plug):
	node : Particle = None
	pass
class LastVelocityPlug(Plug):
	node : Particle = None
	pass
class LastWorldMatrixPlug(Plug):
	node : Particle = None
	pass
class LastWorldPositionPlug(Plug):
	node : Particle = None
	pass
class LastWorldVelocityPlug(Plug):
	node : Particle = None
	pass
class LevelOfDetailPlug(Plug):
	node : Particle = None
	pass
class LifespanModePlug(Plug):
	node : Particle = None
	pass
class LifespanRandomPlug(Plug):
	node : Particle = None
	pass
class MassPlug(Plug):
	node : Particle = None
	pass
class Mass0Plug(Plug):
	node : Particle = None
	pass
class MassCachePlug(Plug):
	node : Particle = None
	pass
class MaxCountPlug(Plug):
	node : Particle = None
	pass
class NetEmittedLastTimePlug(Plug):
	node : Particle = None
	pass
class NewFileFormatPlug(Plug):
	node : Particle = None
	pass
class NewParticlesPlug(Plug):
	node : Particle = None
	pass
class NextIdPlug(Plug):
	node : Particle = None
	pass
class NextId0Plug(Plug):
	node : Particle = None
	pass
class NormalizeVelocityPlug(Plug):
	node : Particle = None
	pass
class NumberOfEventsPlug(Plug):
	node : Particle = None
	pass
class OutputPlug(Plug):
	node : Particle = None
	pass
class OwnerPPFieldDataPlug(Plug):
	node : Particle = None
	pass
class ParentMatrixDirtyPlug(Plug):
	node : Particle = None
	pass
class ParticleIdPlug(Plug):
	node : Particle = None
	pass
class ParticleId0Plug(Plug):
	node : Particle = None
	pass
class ParticleRenderTypePlug(Plug):
	node : Particle = None
	pass
class PositionPlug(Plug):
	node : Particle = None
	pass
class Position0Plug(Plug):
	node : Particle = None
	pass
class PpFieldDataPlug(Plug):
	node : Particle = None
	pass
class RampAccelerationPlug(Plug):
	node : Particle = None
	pass
class RampPositionPlug(Plug):
	node : Particle = None
	pass
class RampVelocityPlug(Plug):
	node : Particle = None
	pass
class RandStateXPlug(Plug):
	parent : RandStatePlug = PlugDescriptor("randState")
	node : Particle = None
	pass
class RandStateYPlug(Plug):
	parent : RandStatePlug = PlugDescriptor("randState")
	node : Particle = None
	pass
class RandStateZPlug(Plug):
	parent : RandStatePlug = PlugDescriptor("randState")
	node : Particle = None
	pass
class RandStatePlug(Plug):
	randStateX_ : RandStateXPlug = PlugDescriptor("randStateX")
	rstx_ : RandStateXPlug = PlugDescriptor("randStateX")
	randStateY_ : RandStateYPlug = PlugDescriptor("randStateY")
	rsty_ : RandStateYPlug = PlugDescriptor("randStateY")
	randStateZ_ : RandStateZPlug = PlugDescriptor("randStateZ")
	rstz_ : RandStateZPlug = PlugDescriptor("randStateZ")
	node : Particle = None
	pass
class SamplerPerParticleDataPlug(Plug):
	node : Particle = None
	pass
class SceneTimeStepSizePlug(Plug):
	node : Particle = None
	pass
class SeedPlug(Plug):
	node : Particle = None
	pass
class ShapeNameMsgPlug(Plug):
	node : Particle = None
	pass
class StartEmittedIndexPlug(Plug):
	node : Particle = None
	pass
class StartFramePlug(Plug):
	node : Particle = None
	pass
class StartTimePlug(Plug):
	node : Particle = None
	pass
class StartupCacheFramePlug(Plug):
	node : Particle = None
	pass
class StartupCachePathPlug(Plug):
	node : Particle = None
	pass
class TargetGeometryPlug(Plug):
	node : Particle = None
	pass
class TargetGeometrySpacePlug(Plug):
	node : Particle = None
	pass
class TargetGeometryWorldMatrixPlug(Plug):
	node : Particle = None
	pass
class TimePlug(Plug):
	node : Particle = None
	pass
class TimeLastComputedPlug(Plug):
	node : Particle = None
	pass
class TimeStepSizePlug(Plug):
	node : Particle = None
	pass
class TotalEventCountPlug(Plug):
	node : Particle = None
	pass
class TraceDepthPlug(Plug):
	node : Particle = None
	pass
class UseCustomCachePlug(Plug):
	node : Particle = None
	pass
class UseStartupCachePlug(Plug):
	node : Particle = None
	pass
class VelocityPlug(Plug):
	node : Particle = None
	pass
class Velocity0Plug(Plug):
	node : Particle = None
	pass
class WorldCentroidXPlug(Plug):
	parent : WorldCentroidPlug = PlugDescriptor("worldCentroid")
	node : Particle = None
	pass
class WorldCentroidYPlug(Plug):
	parent : WorldCentroidPlug = PlugDescriptor("worldCentroid")
	node : Particle = None
	pass
class WorldCentroidZPlug(Plug):
	parent : WorldCentroidPlug = PlugDescriptor("worldCentroid")
	node : Particle = None
	pass
class WorldCentroidPlug(Plug):
	worldCentroidX_ : WorldCentroidXPlug = PlugDescriptor("worldCentroidX")
	wctx_ : WorldCentroidXPlug = PlugDescriptor("worldCentroidX")
	worldCentroidY_ : WorldCentroidYPlug = PlugDescriptor("worldCentroidY")
	wcty_ : WorldCentroidYPlug = PlugDescriptor("worldCentroidY")
	worldCentroidZ_ : WorldCentroidZPlug = PlugDescriptor("worldCentroidZ")
	wctz_ : WorldCentroidZPlug = PlugDescriptor("worldCentroidZ")
	node : Particle = None
	pass
class WorldPositionPlug(Plug):
	node : Particle = None
	pass
class WorldVelocityPlug(Plug):
	node : Particle = None
	pass
class WorldVelocityInObjectSpacePlug(Plug):
	node : Particle = None
	pass
# endregion


# define node class
class Particle(DeformableShape):
	acceleration_ : AccelerationPlug = PlugDescriptor("acceleration")
	acceleration0_ : Acceleration0Plug = PlugDescriptor("acceleration0")
	age_ : AgePlug = PlugDescriptor("age")
	age0_ : Age0Plug = PlugDescriptor("age0")
	ageCache_ : AgeCachePlug = PlugDescriptor("ageCache")
	agesLastDone_ : AgesLastDonePlug = PlugDescriptor("agesLastDone")
	auxiliariesOwned_ : AuxiliariesOwnedPlug = PlugDescriptor("auxiliariesOwned")
	birthTime_ : BirthTimePlug = PlugDescriptor("birthTime")
	birthTime0_ : BirthTime0Plug = PlugDescriptor("birthTime0")
	birthTimeCache_ : BirthTimeCachePlug = PlugDescriptor("birthTimeCache")
	cacheData_ : CacheDataPlug = PlugDescriptor("cacheData")
	cacheWidth_ : CacheWidthPlug = PlugDescriptor("cacheWidth")
	cachedPosition_ : CachedPositionPlug = PlugDescriptor("cachedPosition")
	cachedTime_ : CachedTimePlug = PlugDescriptor("cachedTime")
	cachedVelocity_ : CachedVelocityPlug = PlugDescriptor("cachedVelocity")
	cachedWorldCentroidX_ : CachedWorldCentroidXPlug = PlugDescriptor("cachedWorldCentroidX")
	cachedWorldCentroidY_ : CachedWorldCentroidYPlug = PlugDescriptor("cachedWorldCentroidY")
	cachedWorldCentroidZ_ : CachedWorldCentroidZPlug = PlugDescriptor("cachedWorldCentroidZ")
	cachedWorldCentroid_ : CachedWorldCentroidPlug = PlugDescriptor("cachedWorldCentroid")
	cachedWorldPosition_ : CachedWorldPositionPlug = PlugDescriptor("cachedWorldPosition")
	cachedWorldVelocity_ : CachedWorldVelocityPlug = PlugDescriptor("cachedWorldVelocity")
	centroidX_ : CentroidXPlug = PlugDescriptor("centroidX")
	centroidY_ : CentroidYPlug = PlugDescriptor("centroidY")
	centroidZ_ : CentroidZPlug = PlugDescriptor("centroidZ")
	centroid_ : CentroidPlug = PlugDescriptor("centroid")
	collisionConnections_ : CollisionConnectionsPlug = PlugDescriptor("collisionConnections")
	collisionFriction_ : CollisionFrictionPlug = PlugDescriptor("collisionFriction")
	collisionGeometry_ : CollisionGeometryPlug = PlugDescriptor("collisionGeometry")
	collisionOffset_ : CollisionOffsetPlug = PlugDescriptor("collisionOffset")
	collisionResilience_ : CollisionResiliencePlug = PlugDescriptor("collisionResilience")
	collisionData_ : CollisionDataPlug = PlugDescriptor("collisionData")
	collisionEvents_ : CollisionEventsPlug = PlugDescriptor("collisionEvents")
	collisionRecords_ : CollisionRecordsPlug = PlugDescriptor("collisionRecords")
	collisions_ : CollisionsPlug = PlugDescriptor("collisions")
	computingCount_ : ComputingCountPlug = PlugDescriptor("computingCount")
	connectionsToMe_ : ConnectionsToMePlug = PlugDescriptor("connectionsToMe")
	conserve_ : ConservePlug = PlugDescriptor("conserve")
	count_ : CountPlug = PlugDescriptor("count")
	currentParticle_ : CurrentParticlePlug = PlugDescriptor("currentParticle")
	currentSceneTime_ : CurrentSceneTimePlug = PlugDescriptor("currentSceneTime")
	currentTime_ : CurrentTimePlug = PlugDescriptor("currentTime")
	currentTimeSave_ : CurrentTimeSavePlug = PlugDescriptor("currentTimeSave")
	death_ : DeathPlug = PlugDescriptor("death")
	debugDraw_ : DebugDrawPlug = PlugDescriptor("debugDraw")
	deformedPosition_ : DeformedPositionPlug = PlugDescriptor("deformedPosition")
	depthSort_ : DepthSortPlug = PlugDescriptor("depthSort")
	dieOnEmissionVolumeExit_ : DieOnEmissionVolumeExitPlug = PlugDescriptor("dieOnEmissionVolumeExit")
	diedLastTime_ : DiedLastTimePlug = PlugDescriptor("diedLastTime")
	disableCloudAxis_ : DisableCloudAxisPlug = PlugDescriptor("disableCloudAxis")
	doAge_ : DoAgePlug = PlugDescriptor("doAge")
	doDynamics_ : DoDynamicsPlug = PlugDescriptor("doDynamics")
	doEmission_ : DoEmissionPlug = PlugDescriptor("doEmission")
	dynamicsWeight_ : DynamicsWeightPlug = PlugDescriptor("dynamicsWeight")
	emission_ : EmissionPlug = PlugDescriptor("emission")
	emissionInWorld_ : EmissionInWorldPlug = PlugDescriptor("emissionInWorld")
	emitterConnections_ : EmitterConnectionsPlug = PlugDescriptor("emitterConnections")
	emitterDataDeltaTime_ : EmitterDataDeltaTimePlug = PlugDescriptor("emitterDataDeltaTime")
	emitterDataPosition_ : EmitterDataPositionPlug = PlugDescriptor("emitterDataPosition")
	emitterDataVelocity_ : EmitterDataVelocityPlug = PlugDescriptor("emitterDataVelocity")
	emitterData_ : EmitterDataPlug = PlugDescriptor("emitterData")
	emitterId_ : EmitterIdPlug = PlugDescriptor("emitterId")
	emitterId0_ : EmitterId0Plug = PlugDescriptor("emitterId0")
	enforceCountFromHistory_ : EnforceCountFromHistoryPlug = PlugDescriptor("enforceCountFromHistory")
	evaluationTime_ : EvaluationTimePlug = PlugDescriptor("evaluationTime")
	eventCount_ : EventCountPlug = PlugDescriptor("eventCount")
	eventDie_ : EventDiePlug = PlugDescriptor("eventDie")
	eventEmit_ : EventEmitPlug = PlugDescriptor("eventEmit")
	eventName_ : EventNamePlug = PlugDescriptor("eventName")
	eventNameCount_ : EventNameCountPlug = PlugDescriptor("eventNameCount")
	eventProc_ : EventProcPlug = PlugDescriptor("eventProc")
	eventRandStateX_ : EventRandStateXPlug = PlugDescriptor("eventRandStateX")
	eventRandStateY_ : EventRandStateYPlug = PlugDescriptor("eventRandStateY")
	eventRandStateZ_ : EventRandStateZPlug = PlugDescriptor("eventRandStateZ")
	eventRandState_ : EventRandStatePlug = PlugDescriptor("eventRandState")
	eventRandom_ : EventRandomPlug = PlugDescriptor("eventRandom")
	eventSeed_ : EventSeedPlug = PlugDescriptor("eventSeed")
	eventSplit_ : EventSplitPlug = PlugDescriptor("eventSplit")
	eventSpread_ : EventSpreadPlug = PlugDescriptor("eventSpread")
	eventTarget_ : EventTargetPlug = PlugDescriptor("eventTarget")
	eventTest_ : EventTestPlug = PlugDescriptor("eventTest")
	eventValid_ : EventValidPlug = PlugDescriptor("eventValid")
	executeCreationExpression_ : ExecuteCreationExpressionPlug = PlugDescriptor("executeCreationExpression")
	executeRuntimeAfterDynamicsExpression_ : ExecuteRuntimeAfterDynamicsExpressionPlug = PlugDescriptor("executeRuntimeAfterDynamicsExpression")
	executeRuntimeBeforeDynamicsExpression_ : ExecuteRuntimeBeforeDynamicsExpressionPlug = PlugDescriptor("executeRuntimeBeforeDynamicsExpression")
	expressionsAfterDynamics_ : ExpressionsAfterDynamicsPlug = PlugDescriptor("expressionsAfterDynamics")
	fieldConnections_ : FieldConnectionsPlug = PlugDescriptor("fieldConnections")
	fieldDataDeltaTime_ : FieldDataDeltaTimePlug = PlugDescriptor("fieldDataDeltaTime")
	fieldDataMass_ : FieldDataMassPlug = PlugDescriptor("fieldDataMass")
	fieldDataPosition_ : FieldDataPositionPlug = PlugDescriptor("fieldDataPosition")
	fieldDataVelocity_ : FieldDataVelocityPlug = PlugDescriptor("fieldDataVelocity")
	fieldData_ : FieldDataPlug = PlugDescriptor("fieldData")
	finalLifespanPP_ : FinalLifespanPPPlug = PlugDescriptor("finalLifespanPP")
	force_ : ForcePlug = PlugDescriptor("force")
	forceDynamics_ : ForceDynamicsPlug = PlugDescriptor("forceDynamics")
	forceEmission_ : ForceEmissionPlug = PlugDescriptor("forceEmission")
	forcesInWorld_ : ForcesInWorldPlug = PlugDescriptor("forcesInWorld")
	frame_ : FramePlug = PlugDescriptor("frame")
	generalSeed_ : GeneralSeedPlug = PlugDescriptor("generalSeed")
	goalActive_ : GoalActivePlug = PlugDescriptor("goalActive")
	goalGeometry_ : GoalGeometryPlug = PlugDescriptor("goalGeometry")
	goalSmoothness_ : GoalSmoothnessPlug = PlugDescriptor("goalSmoothness")
	goalUvSetName_ : GoalUvSetNamePlug = PlugDescriptor("goalUvSetName")
	goalWeight_ : GoalWeightPlug = PlugDescriptor("goalWeight")
	idCache_ : IdCachePlug = PlugDescriptor("idCache")
	idIndex_ : IdIndexPlug = PlugDescriptor("idIndex")
	sortedId_ : SortedIdPlug = PlugDescriptor("sortedId")
	idMapping_ : IdMappingPlug = PlugDescriptor("idMapping")
	inheritColor_ : InheritColorPlug = PlugDescriptor("inheritColor")
	inheritFactor_ : InheritFactorPlug = PlugDescriptor("inheritFactor")
	input_ : InputPlug = PlugDescriptor("input")
	inputForce_ : InputForcePlug = PlugDescriptor("inputForce")
	inputGeometry_ : InputGeometryPlug = PlugDescriptor("inputGeometry")
	inputGeometryPoints_ : InputGeometryPointsPlug = PlugDescriptor("inputGeometryPoints")
	inputGeometrySpace_ : InputGeometrySpacePlug = PlugDescriptor("inputGeometrySpace")
	instanceAttributeMapping_ : InstanceAttributeMappingPlug = PlugDescriptor("instanceAttributeMapping")
	instancePointData_ : InstancePointDataPlug = PlugDescriptor("instancePointData")
	instanceData_ : InstanceDataPlug = PlugDescriptor("instanceData")
	internalCreationExpression_ : InternalCreationExpressionPlug = PlugDescriptor("internalCreationExpression")
	internalRuntimeAfterDynamicsExpression_ : InternalRuntimeAfterDynamicsExpressionPlug = PlugDescriptor("internalRuntimeAfterDynamicsExpression")
	internalRuntimeBeforeDynamicsExpression_ : InternalRuntimeBeforeDynamicsExpressionPlug = PlugDescriptor("internalRuntimeBeforeDynamicsExpression")
	internalRuntimeExpression_ : InternalRuntimeExpressionPlug = PlugDescriptor("internalRuntimeExpression")
	isDynamic_ : IsDynamicPlug = PlugDescriptor("isDynamic")
	isFull_ : IsFullPlug = PlugDescriptor("isFull")
	lastCachedPosition_ : LastCachedPositionPlug = PlugDescriptor("lastCachedPosition")
	lastPosition_ : LastPositionPlug = PlugDescriptor("lastPosition")
	lastSceneTime_ : LastSceneTimePlug = PlugDescriptor("lastSceneTime")
	lastTimeEvaluated_ : LastTimeEvaluatedPlug = PlugDescriptor("lastTimeEvaluated")
	lastTotalEventCount_ : LastTotalEventCountPlug = PlugDescriptor("lastTotalEventCount")
	lastVelocity_ : LastVelocityPlug = PlugDescriptor("lastVelocity")
	lastWorldMatrix_ : LastWorldMatrixPlug = PlugDescriptor("lastWorldMatrix")
	lastWorldPosition_ : LastWorldPositionPlug = PlugDescriptor("lastWorldPosition")
	lastWorldVelocity_ : LastWorldVelocityPlug = PlugDescriptor("lastWorldVelocity")
	levelOfDetail_ : LevelOfDetailPlug = PlugDescriptor("levelOfDetail")
	lifespanMode_ : LifespanModePlug = PlugDescriptor("lifespanMode")
	lifespanRandom_ : LifespanRandomPlug = PlugDescriptor("lifespanRandom")
	mass_ : MassPlug = PlugDescriptor("mass")
	mass0_ : Mass0Plug = PlugDescriptor("mass0")
	massCache_ : MassCachePlug = PlugDescriptor("massCache")
	maxCount_ : MaxCountPlug = PlugDescriptor("maxCount")
	netEmittedLastTime_ : NetEmittedLastTimePlug = PlugDescriptor("netEmittedLastTime")
	newFileFormat_ : NewFileFormatPlug = PlugDescriptor("newFileFormat")
	newParticles_ : NewParticlesPlug = PlugDescriptor("newParticles")
	nextId_ : NextIdPlug = PlugDescriptor("nextId")
	nextId0_ : NextId0Plug = PlugDescriptor("nextId0")
	normalizeVelocity_ : NormalizeVelocityPlug = PlugDescriptor("normalizeVelocity")
	numberOfEvents_ : NumberOfEventsPlug = PlugDescriptor("numberOfEvents")
	output_ : OutputPlug = PlugDescriptor("output")
	ownerPPFieldData_ : OwnerPPFieldDataPlug = PlugDescriptor("ownerPPFieldData")
	parentMatrixDirty_ : ParentMatrixDirtyPlug = PlugDescriptor("parentMatrixDirty")
	particleId_ : ParticleIdPlug = PlugDescriptor("particleId")
	particleId0_ : ParticleId0Plug = PlugDescriptor("particleId0")
	particleRenderType_ : ParticleRenderTypePlug = PlugDescriptor("particleRenderType")
	position_ : PositionPlug = PlugDescriptor("position")
	position0_ : Position0Plug = PlugDescriptor("position0")
	ppFieldData_ : PpFieldDataPlug = PlugDescriptor("ppFieldData")
	rampAcceleration_ : RampAccelerationPlug = PlugDescriptor("rampAcceleration")
	rampPosition_ : RampPositionPlug = PlugDescriptor("rampPosition")
	rampVelocity_ : RampVelocityPlug = PlugDescriptor("rampVelocity")
	randStateX_ : RandStateXPlug = PlugDescriptor("randStateX")
	randStateY_ : RandStateYPlug = PlugDescriptor("randStateY")
	randStateZ_ : RandStateZPlug = PlugDescriptor("randStateZ")
	randState_ : RandStatePlug = PlugDescriptor("randState")
	samplerPerParticleData_ : SamplerPerParticleDataPlug = PlugDescriptor("samplerPerParticleData")
	sceneTimeStepSize_ : SceneTimeStepSizePlug = PlugDescriptor("sceneTimeStepSize")
	seed_ : SeedPlug = PlugDescriptor("seed")
	shapeNameMsg_ : ShapeNameMsgPlug = PlugDescriptor("shapeNameMsg")
	startEmittedIndex_ : StartEmittedIndexPlug = PlugDescriptor("startEmittedIndex")
	startFrame_ : StartFramePlug = PlugDescriptor("startFrame")
	startTime_ : StartTimePlug = PlugDescriptor("startTime")
	startupCacheFrame_ : StartupCacheFramePlug = PlugDescriptor("startupCacheFrame")
	startupCachePath_ : StartupCachePathPlug = PlugDescriptor("startupCachePath")
	targetGeometry_ : TargetGeometryPlug = PlugDescriptor("targetGeometry")
	targetGeometrySpace_ : TargetGeometrySpacePlug = PlugDescriptor("targetGeometrySpace")
	targetGeometryWorldMatrix_ : TargetGeometryWorldMatrixPlug = PlugDescriptor("targetGeometryWorldMatrix")
	time_ : TimePlug = PlugDescriptor("time")
	timeLastComputed_ : TimeLastComputedPlug = PlugDescriptor("timeLastComputed")
	timeStepSize_ : TimeStepSizePlug = PlugDescriptor("timeStepSize")
	totalEventCount_ : TotalEventCountPlug = PlugDescriptor("totalEventCount")
	traceDepth_ : TraceDepthPlug = PlugDescriptor("traceDepth")
	useCustomCache_ : UseCustomCachePlug = PlugDescriptor("useCustomCache")
	useStartupCache_ : UseStartupCachePlug = PlugDescriptor("useStartupCache")
	velocity_ : VelocityPlug = PlugDescriptor("velocity")
	velocity0_ : Velocity0Plug = PlugDescriptor("velocity0")
	worldCentroidX_ : WorldCentroidXPlug = PlugDescriptor("worldCentroidX")
	worldCentroidY_ : WorldCentroidYPlug = PlugDescriptor("worldCentroidY")
	worldCentroidZ_ : WorldCentroidZPlug = PlugDescriptor("worldCentroidZ")
	worldCentroid_ : WorldCentroidPlug = PlugDescriptor("worldCentroid")
	worldPosition_ : WorldPositionPlug = PlugDescriptor("worldPosition")
	worldVelocity_ : WorldVelocityPlug = PlugDescriptor("worldVelocity")
	worldVelocityInObjectSpace_ : WorldVelocityInObjectSpacePlug = PlugDescriptor("worldVelocityInObjectSpace")

	# node attributes

	typeName = "particle"
	apiTypeInt = 311
	apiTypeStr = "kParticle"
	typeIdInt = 1498431826
	MFnCls = om.MFnDagNode
	pass

