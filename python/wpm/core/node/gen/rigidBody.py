

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Shape = retriever.getNodeCls("Shape")
assert Shape
if T.TYPE_CHECKING:
	from .. import Shape

# add node doc



# region plug type defs
class ActivePlug(Plug):
	node : RigidBody = None
	pass
class AllowDisconnectionPlug(Plug):
	node : RigidBody = None
	pass
class ApplyForceAtPlug(Plug):
	node : RigidBody = None
	pass
class AutoInitPlug(Plug):
	node : RigidBody = None
	pass
class BakeSimulationIndexPlug(Plug):
	node : RigidBody = None
	pass
class BouncinessPlug(Plug):
	node : RigidBody = None
	pass
class CacheDataPlug(Plug):
	node : RigidBody = None
	pass
class CacheDirtyArrayPlug(Plug):
	node : RigidBody = None
	pass
class CachedFrameCountPlug(Plug):
	node : RigidBody = None
	pass
class CenterOfMassXPlug(Plug):
	parent : CenterOfMassPlug = PlugDescriptor("centerOfMass")
	node : RigidBody = None
	pass
class CenterOfMassYPlug(Plug):
	parent : CenterOfMassPlug = PlugDescriptor("centerOfMass")
	node : RigidBody = None
	pass
class CenterOfMassZPlug(Plug):
	parent : CenterOfMassPlug = PlugDescriptor("centerOfMass")
	node : RigidBody = None
	pass
class CenterOfMassPlug(Plug):
	centerOfMassX_ : CenterOfMassXPlug = PlugDescriptor("centerOfMassX")
	cmx_ : CenterOfMassXPlug = PlugDescriptor("centerOfMassX")
	centerOfMassY_ : CenterOfMassYPlug = PlugDescriptor("centerOfMassY")
	cmy_ : CenterOfMassYPlug = PlugDescriptor("centerOfMassY")
	centerOfMassZ_ : CenterOfMassZPlug = PlugDescriptor("centerOfMassZ")
	cmz_ : CenterOfMassZPlug = PlugDescriptor("centerOfMassZ")
	node : RigidBody = None
	pass
class ChoicePlug(Plug):
	node : RigidBody = None
	pass
class CollisionLayerPlug(Plug):
	node : RigidBody = None
	pass
class CollisionRecordsPlug(Plug):
	node : RigidBody = None
	pass
class CollisionsPlug(Plug):
	node : RigidBody = None
	pass
class ContactCountPlug(Plug):
	node : RigidBody = None
	pass
class ContactNamePlug(Plug):
	node : RigidBody = None
	pass
class ContactXPlug(Plug):
	parent : ContactPositionPlug = PlugDescriptor("contactPosition")
	node : RigidBody = None
	pass
class ContactYPlug(Plug):
	parent : ContactPositionPlug = PlugDescriptor("contactPosition")
	node : RigidBody = None
	pass
class ContactZPlug(Plug):
	parent : ContactPositionPlug = PlugDescriptor("contactPosition")
	node : RigidBody = None
	pass
class ContactPositionPlug(Plug):
	contactX_ : ContactXPlug = PlugDescriptor("contactX")
	cnx_ : ContactXPlug = PlugDescriptor("contactX")
	contactY_ : ContactYPlug = PlugDescriptor("contactY")
	cny_ : ContactYPlug = PlugDescriptor("contactY")
	contactZ_ : ContactZPlug = PlugDescriptor("contactZ")
	cnz_ : ContactZPlug = PlugDescriptor("contactZ")
	node : RigidBody = None
	pass
class CurrentTimePlug(Plug):
	node : RigidBody = None
	pass
class DampingPlug(Plug):
	node : RigidBody = None
	pass
class DataCachePlug(Plug):
	node : RigidBody = None
	pass
class DebugDrawPlug(Plug):
	node : RigidBody = None
	pass
class DynamicFrictionPlug(Plug):
	node : RigidBody = None
	pass
class FieldConnectionsPlug(Plug):
	node : RigidBody = None
	pass
class DeltaTimePlug(Plug):
	parent : FieldDataPlug = PlugDescriptor("fieldData")
	node : RigidBody = None
	pass
class FieldDataMassPlug(Plug):
	parent : FieldDataPlug = PlugDescriptor("fieldData")
	node : RigidBody = None
	pass
class FieldDataPositionPlug(Plug):
	parent : FieldDataPlug = PlugDescriptor("fieldData")
	node : RigidBody = None
	pass
class FieldDataVelocityPlug(Plug):
	parent : FieldDataPlug = PlugDescriptor("fieldData")
	node : RigidBody = None
	pass
class FieldDataPlug(Plug):
	deltaTime_ : DeltaTimePlug = PlugDescriptor("deltaTime")
	dt_ : DeltaTimePlug = PlugDescriptor("deltaTime")
	fieldDataMass_ : FieldDataMassPlug = PlugDescriptor("fieldDataMass")
	fdm_ : FieldDataMassPlug = PlugDescriptor("fieldDataMass")
	fieldDataPosition_ : FieldDataPositionPlug = PlugDescriptor("fieldDataPosition")
	fdp_ : FieldDataPositionPlug = PlugDescriptor("fieldDataPosition")
	fieldDataVelocity_ : FieldDataVelocityPlug = PlugDescriptor("fieldDataVelocity")
	fdv_ : FieldDataVelocityPlug = PlugDescriptor("fieldDataVelocity")
	node : RigidBody = None
	pass
class FirstCachedFramePlug(Plug):
	node : RigidBody = None
	pass
class ForceXPlug(Plug):
	parent : ForcePlug = PlugDescriptor("force")
	node : RigidBody = None
	pass
class ForceYPlug(Plug):
	parent : ForcePlug = PlugDescriptor("force")
	node : RigidBody = None
	pass
class ForceZPlug(Plug):
	parent : ForcePlug = PlugDescriptor("force")
	node : RigidBody = None
	pass
class ForcePlug(Plug):
	forceX_ : ForceXPlug = PlugDescriptor("forceX")
	fx_ : ForceXPlug = PlugDescriptor("forceX")
	forceY_ : ForceYPlug = PlugDescriptor("forceY")
	fy_ : ForceYPlug = PlugDescriptor("forceY")
	forceZ_ : ForceZPlug = PlugDescriptor("forceZ")
	fz_ : ForceZPlug = PlugDescriptor("forceZ")
	node : RigidBody = None
	pass
class OutputForcePlug(Plug):
	parent : GeneralForcePlug = PlugDescriptor("generalForce")
	node : RigidBody = None
	pass
class OutputTorquePlug(Plug):
	parent : GeneralForcePlug = PlugDescriptor("generalForce")
	node : RigidBody = None
	pass
class GeneralForcePlug(Plug):
	outputForce_ : OutputForcePlug = PlugDescriptor("outputForce")
	ofr_ : OutputForcePlug = PlugDescriptor("outputForce")
	outputTorque_ : OutputTorquePlug = PlugDescriptor("outputTorque")
	otr_ : OutputTorquePlug = PlugDescriptor("outputTorque")
	node : RigidBody = None
	pass
class IgnorePlug(Plug):
	node : RigidBody = None
	pass
class ImpulseXPlug(Plug):
	parent : ImpulsePlug = PlugDescriptor("impulse")
	node : RigidBody = None
	pass
class ImpulseYPlug(Plug):
	parent : ImpulsePlug = PlugDescriptor("impulse")
	node : RigidBody = None
	pass
class ImpulseZPlug(Plug):
	parent : ImpulsePlug = PlugDescriptor("impulse")
	node : RigidBody = None
	pass
class ImpulsePlug(Plug):
	impulseX_ : ImpulseXPlug = PlugDescriptor("impulseX")
	imx_ : ImpulseXPlug = PlugDescriptor("impulseX")
	impulseY_ : ImpulseYPlug = PlugDescriptor("impulseY")
	imy_ : ImpulseYPlug = PlugDescriptor("impulseY")
	impulseZ_ : ImpulseZPlug = PlugDescriptor("impulseZ")
	imz_ : ImpulseZPlug = PlugDescriptor("impulseZ")
	node : RigidBody = None
	pass
class ImpulsePositionXPlug(Plug):
	parent : ImpulsePositionPlug = PlugDescriptor("impulsePosition")
	node : RigidBody = None
	pass
class ImpulsePositionYPlug(Plug):
	parent : ImpulsePositionPlug = PlugDescriptor("impulsePosition")
	node : RigidBody = None
	pass
class ImpulsePositionZPlug(Plug):
	parent : ImpulsePositionPlug = PlugDescriptor("impulsePosition")
	node : RigidBody = None
	pass
class ImpulsePositionPlug(Plug):
	impulsePositionX_ : ImpulsePositionXPlug = PlugDescriptor("impulsePositionX")
	pix_ : ImpulsePositionXPlug = PlugDescriptor("impulsePositionX")
	impulsePositionY_ : ImpulsePositionYPlug = PlugDescriptor("impulsePositionY")
	piy_ : ImpulsePositionYPlug = PlugDescriptor("impulsePositionY")
	impulsePositionZ_ : ImpulsePositionZPlug = PlugDescriptor("impulsePositionZ")
	piz_ : ImpulsePositionZPlug = PlugDescriptor("impulsePositionZ")
	node : RigidBody = None
	pass
class InitialOrientationXPlug(Plug):
	parent : InitialOrientationPlug = PlugDescriptor("initialOrientation")
	node : RigidBody = None
	pass
class InitialOrientationYPlug(Plug):
	parent : InitialOrientationPlug = PlugDescriptor("initialOrientation")
	node : RigidBody = None
	pass
class InitialOrientationZPlug(Plug):
	parent : InitialOrientationPlug = PlugDescriptor("initialOrientation")
	node : RigidBody = None
	pass
class InitialOrientationPlug(Plug):
	initialOrientationX_ : InitialOrientationXPlug = PlugDescriptor("initialOrientationX")
	iox_ : InitialOrientationXPlug = PlugDescriptor("initialOrientationX")
	initialOrientationY_ : InitialOrientationYPlug = PlugDescriptor("initialOrientationY")
	ioy_ : InitialOrientationYPlug = PlugDescriptor("initialOrientationY")
	initialOrientationZ_ : InitialOrientationZPlug = PlugDescriptor("initialOrientationZ")
	ioz_ : InitialOrientationZPlug = PlugDescriptor("initialOrientationZ")
	node : RigidBody = None
	pass
class InitialPositionXPlug(Plug):
	parent : InitialPositionPlug = PlugDescriptor("initialPosition")
	node : RigidBody = None
	pass
class InitialPositionYPlug(Plug):
	parent : InitialPositionPlug = PlugDescriptor("initialPosition")
	node : RigidBody = None
	pass
class InitialPositionZPlug(Plug):
	parent : InitialPositionPlug = PlugDescriptor("initialPosition")
	node : RigidBody = None
	pass
class InitialPositionPlug(Plug):
	initialPositionX_ : InitialPositionXPlug = PlugDescriptor("initialPositionX")
	ipx_ : InitialPositionXPlug = PlugDescriptor("initialPositionX")
	initialPositionY_ : InitialPositionYPlug = PlugDescriptor("initialPositionY")
	ipy_ : InitialPositionYPlug = PlugDescriptor("initialPositionY")
	initialPositionZ_ : InitialPositionZPlug = PlugDescriptor("initialPositionZ")
	ipz_ : InitialPositionZPlug = PlugDescriptor("initialPositionZ")
	node : RigidBody = None
	pass
class InitialSpinXPlug(Plug):
	parent : InitialSpinPlug = PlugDescriptor("initialSpin")
	node : RigidBody = None
	pass
class InitialSpinYPlug(Plug):
	parent : InitialSpinPlug = PlugDescriptor("initialSpin")
	node : RigidBody = None
	pass
class InitialSpinZPlug(Plug):
	parent : InitialSpinPlug = PlugDescriptor("initialSpin")
	node : RigidBody = None
	pass
class InitialSpinPlug(Plug):
	initialSpinX_ : InitialSpinXPlug = PlugDescriptor("initialSpinX")
	isx_ : InitialSpinXPlug = PlugDescriptor("initialSpinX")
	initialSpinY_ : InitialSpinYPlug = PlugDescriptor("initialSpinY")
	isy_ : InitialSpinYPlug = PlugDescriptor("initialSpinY")
	initialSpinZ_ : InitialSpinZPlug = PlugDescriptor("initialSpinZ")
	isz_ : InitialSpinZPlug = PlugDescriptor("initialSpinZ")
	node : RigidBody = None
	pass
class InitialVelocityXPlug(Plug):
	parent : InitialVelocityPlug = PlugDescriptor("initialVelocity")
	node : RigidBody = None
	pass
class InitialVelocityYPlug(Plug):
	parent : InitialVelocityPlug = PlugDescriptor("initialVelocity")
	node : RigidBody = None
	pass
class InitialVelocityZPlug(Plug):
	parent : InitialVelocityPlug = PlugDescriptor("initialVelocity")
	node : RigidBody = None
	pass
class InitialVelocityPlug(Plug):
	initialVelocityX_ : InitialVelocityXPlug = PlugDescriptor("initialVelocityX")
	ivx_ : InitialVelocityXPlug = PlugDescriptor("initialVelocityX")
	initialVelocityY_ : InitialVelocityYPlug = PlugDescriptor("initialVelocityY")
	ivy_ : InitialVelocityYPlug = PlugDescriptor("initialVelocityY")
	initialVelocityZ_ : InitialVelocityZPlug = PlugDescriptor("initialVelocityZ")
	ivz_ : InitialVelocityZPlug = PlugDescriptor("initialVelocityZ")
	node : RigidBody = None
	pass
class InputForcePlug(Plug):
	node : RigidBody = None
	pass
class InputForceTypePlug(Plug):
	node : RigidBody = None
	pass
class InputGeometryCntPlug(Plug):
	node : RigidBody = None
	pass
class InputGeometryMsgPlug(Plug):
	node : RigidBody = None
	pass
class InterpenetrateWithPlug(Plug):
	node : RigidBody = None
	pass
class IsKeyframedPlug(Plug):
	node : RigidBody = None
	pass
class IsKinematicPlug(Plug):
	node : RigidBody = None
	pass
class IsParentedPlug(Plug):
	node : RigidBody = None
	pass
class LastCachedFramePlug(Plug):
	node : RigidBody = None
	pass
class LastPositionXPlug(Plug):
	parent : LastPositionPlug = PlugDescriptor("lastPosition")
	node : RigidBody = None
	pass
class LastPositionYPlug(Plug):
	parent : LastPositionPlug = PlugDescriptor("lastPosition")
	node : RigidBody = None
	pass
class LastPositionZPlug(Plug):
	parent : LastPositionPlug = PlugDescriptor("lastPosition")
	node : RigidBody = None
	pass
class LastPositionPlug(Plug):
	lastPositionX_ : LastPositionXPlug = PlugDescriptor("lastPositionX")
	lpx_ : LastPositionXPlug = PlugDescriptor("lastPositionX")
	lastPositionY_ : LastPositionYPlug = PlugDescriptor("lastPositionY")
	lpy_ : LastPositionYPlug = PlugDescriptor("lastPositionY")
	lastPositionZ_ : LastPositionZPlug = PlugDescriptor("lastPositionZ")
	lpz_ : LastPositionZPlug = PlugDescriptor("lastPositionZ")
	node : RigidBody = None
	pass
class LastRotationXPlug(Plug):
	parent : LastRotationPlug = PlugDescriptor("lastRotation")
	node : RigidBody = None
	pass
class LastRotationYPlug(Plug):
	parent : LastRotationPlug = PlugDescriptor("lastRotation")
	node : RigidBody = None
	pass
class LastRotationZPlug(Plug):
	parent : LastRotationPlug = PlugDescriptor("lastRotation")
	node : RigidBody = None
	pass
class LastRotationPlug(Plug):
	lastRotationX_ : LastRotationXPlug = PlugDescriptor("lastRotationX")
	lrx_ : LastRotationXPlug = PlugDescriptor("lastRotationX")
	lastRotationY_ : LastRotationYPlug = PlugDescriptor("lastRotationY")
	lry_ : LastRotationYPlug = PlugDescriptor("lastRotationY")
	lastRotationZ_ : LastRotationZPlug = PlugDescriptor("lastRotationZ")
	lrz_ : LastRotationZPlug = PlugDescriptor("lastRotationZ")
	node : RigidBody = None
	pass
class LastSceneTimePlug(Plug):
	node : RigidBody = None
	pass
class LockCenterOfMassPlug(Plug):
	node : RigidBody = None
	pass
class MassPlug(Plug):
	node : RigidBody = None
	pass
class ParticleCollisionPlug(Plug):
	node : RigidBody = None
	pass
class RigidWorldMatrixPlug(Plug):
	node : RigidBody = None
	pass
class RunUpCachePlug(Plug):
	node : RigidBody = None
	pass
class ShapeChangedPlug(Plug):
	node : RigidBody = None
	pass
class SolverIdPlug(Plug):
	node : RigidBody = None
	pass
class SpinXPlug(Plug):
	parent : SpinPlug = PlugDescriptor("spin")
	node : RigidBody = None
	pass
class SpinYPlug(Plug):
	parent : SpinPlug = PlugDescriptor("spin")
	node : RigidBody = None
	pass
class SpinZPlug(Plug):
	parent : SpinPlug = PlugDescriptor("spin")
	node : RigidBody = None
	pass
class SpinPlug(Plug):
	spinX_ : SpinXPlug = PlugDescriptor("spinX")
	spx_ : SpinXPlug = PlugDescriptor("spinX")
	spinY_ : SpinYPlug = PlugDescriptor("spinY")
	spy_ : SpinYPlug = PlugDescriptor("spinY")
	spinZ_ : SpinZPlug = PlugDescriptor("spinZ")
	spz_ : SpinZPlug = PlugDescriptor("spinZ")
	node : RigidBody = None
	pass
class SpinImpulseXPlug(Plug):
	parent : SpinImpulsePlug = PlugDescriptor("spinImpulse")
	node : RigidBody = None
	pass
class SpinImpulseYPlug(Plug):
	parent : SpinImpulsePlug = PlugDescriptor("spinImpulse")
	node : RigidBody = None
	pass
class SpinImpulseZPlug(Plug):
	parent : SpinImpulsePlug = PlugDescriptor("spinImpulse")
	node : RigidBody = None
	pass
class SpinImpulsePlug(Plug):
	spinImpulseX_ : SpinImpulseXPlug = PlugDescriptor("spinImpulseX")
	six_ : SpinImpulseXPlug = PlugDescriptor("spinImpulseX")
	spinImpulseY_ : SpinImpulseYPlug = PlugDescriptor("spinImpulseY")
	siy_ : SpinImpulseYPlug = PlugDescriptor("spinImpulseY")
	spinImpulseZ_ : SpinImpulseZPlug = PlugDescriptor("spinImpulseZ")
	siz_ : SpinImpulseZPlug = PlugDescriptor("spinImpulseZ")
	node : RigidBody = None
	pass
class StandInPlug(Plug):
	node : RigidBody = None
	pass
class StaticFrictionPlug(Plug):
	node : RigidBody = None
	pass
class TessellationFactorPlug(Plug):
	node : RigidBody = None
	pass
class TorqueXPlug(Plug):
	parent : TorquePlug = PlugDescriptor("torque")
	node : RigidBody = None
	pass
class TorqueYPlug(Plug):
	parent : TorquePlug = PlugDescriptor("torque")
	node : RigidBody = None
	pass
class TorqueZPlug(Plug):
	parent : TorquePlug = PlugDescriptor("torque")
	node : RigidBody = None
	pass
class TorquePlug(Plug):
	torqueX_ : TorqueXPlug = PlugDescriptor("torqueX")
	trx_ : TorqueXPlug = PlugDescriptor("torqueX")
	torqueY_ : TorqueYPlug = PlugDescriptor("torqueY")
	try_ : TorqueYPlug = PlugDescriptor("torqueY")
	torqueZ_ : TorqueZPlug = PlugDescriptor("torqueZ")
	trz_ : TorqueZPlug = PlugDescriptor("torqueZ")
	node : RigidBody = None
	pass
class VelocityXPlug(Plug):
	parent : VelocityPlug = PlugDescriptor("velocity")
	node : RigidBody = None
	pass
class VelocityYPlug(Plug):
	parent : VelocityPlug = PlugDescriptor("velocity")
	node : RigidBody = None
	pass
class VelocityZPlug(Plug):
	parent : VelocityPlug = PlugDescriptor("velocity")
	node : RigidBody = None
	pass
class VelocityPlug(Plug):
	velocityX_ : VelocityXPlug = PlugDescriptor("velocityX")
	vx_ : VelocityXPlug = PlugDescriptor("velocityX")
	velocityY_ : VelocityYPlug = PlugDescriptor("velocityY")
	vy_ : VelocityYPlug = PlugDescriptor("velocityY")
	velocityZ_ : VelocityZPlug = PlugDescriptor("velocityZ")
	vz_ : VelocityZPlug = PlugDescriptor("velocityZ")
	node : RigidBody = None
	pass
class VolumePlug(Plug):
	node : RigidBody = None
	pass
# endregion


# define node class
class RigidBody(Shape):
	active_ : ActivePlug = PlugDescriptor("active")
	allowDisconnection_ : AllowDisconnectionPlug = PlugDescriptor("allowDisconnection")
	applyForceAt_ : ApplyForceAtPlug = PlugDescriptor("applyForceAt")
	autoInit_ : AutoInitPlug = PlugDescriptor("autoInit")
	bakeSimulationIndex_ : BakeSimulationIndexPlug = PlugDescriptor("bakeSimulationIndex")
	bounciness_ : BouncinessPlug = PlugDescriptor("bounciness")
	cacheData_ : CacheDataPlug = PlugDescriptor("cacheData")
	cacheDirtyArray_ : CacheDirtyArrayPlug = PlugDescriptor("cacheDirtyArray")
	cachedFrameCount_ : CachedFrameCountPlug = PlugDescriptor("cachedFrameCount")
	centerOfMassX_ : CenterOfMassXPlug = PlugDescriptor("centerOfMassX")
	centerOfMassY_ : CenterOfMassYPlug = PlugDescriptor("centerOfMassY")
	centerOfMassZ_ : CenterOfMassZPlug = PlugDescriptor("centerOfMassZ")
	centerOfMass_ : CenterOfMassPlug = PlugDescriptor("centerOfMass")
	choice_ : ChoicePlug = PlugDescriptor("choice")
	collisionLayer_ : CollisionLayerPlug = PlugDescriptor("collisionLayer")
	collisionRecords_ : CollisionRecordsPlug = PlugDescriptor("collisionRecords")
	collisions_ : CollisionsPlug = PlugDescriptor("collisions")
	contactCount_ : ContactCountPlug = PlugDescriptor("contactCount")
	contactName_ : ContactNamePlug = PlugDescriptor("contactName")
	contactX_ : ContactXPlug = PlugDescriptor("contactX")
	contactY_ : ContactYPlug = PlugDescriptor("contactY")
	contactZ_ : ContactZPlug = PlugDescriptor("contactZ")
	contactPosition_ : ContactPositionPlug = PlugDescriptor("contactPosition")
	currentTime_ : CurrentTimePlug = PlugDescriptor("currentTime")
	damping_ : DampingPlug = PlugDescriptor("damping")
	dataCache_ : DataCachePlug = PlugDescriptor("dataCache")
	debugDraw_ : DebugDrawPlug = PlugDescriptor("debugDraw")
	dynamicFriction_ : DynamicFrictionPlug = PlugDescriptor("dynamicFriction")
	fieldConnections_ : FieldConnectionsPlug = PlugDescriptor("fieldConnections")
	deltaTime_ : DeltaTimePlug = PlugDescriptor("deltaTime")
	fieldDataMass_ : FieldDataMassPlug = PlugDescriptor("fieldDataMass")
	fieldDataPosition_ : FieldDataPositionPlug = PlugDescriptor("fieldDataPosition")
	fieldDataVelocity_ : FieldDataVelocityPlug = PlugDescriptor("fieldDataVelocity")
	fieldData_ : FieldDataPlug = PlugDescriptor("fieldData")
	firstCachedFrame_ : FirstCachedFramePlug = PlugDescriptor("firstCachedFrame")
	forceX_ : ForceXPlug = PlugDescriptor("forceX")
	forceY_ : ForceYPlug = PlugDescriptor("forceY")
	forceZ_ : ForceZPlug = PlugDescriptor("forceZ")
	force_ : ForcePlug = PlugDescriptor("force")
	outputForce_ : OutputForcePlug = PlugDescriptor("outputForce")
	outputTorque_ : OutputTorquePlug = PlugDescriptor("outputTorque")
	generalForce_ : GeneralForcePlug = PlugDescriptor("generalForce")
	ignore_ : IgnorePlug = PlugDescriptor("ignore")
	impulseX_ : ImpulseXPlug = PlugDescriptor("impulseX")
	impulseY_ : ImpulseYPlug = PlugDescriptor("impulseY")
	impulseZ_ : ImpulseZPlug = PlugDescriptor("impulseZ")
	impulse_ : ImpulsePlug = PlugDescriptor("impulse")
	impulsePositionX_ : ImpulsePositionXPlug = PlugDescriptor("impulsePositionX")
	impulsePositionY_ : ImpulsePositionYPlug = PlugDescriptor("impulsePositionY")
	impulsePositionZ_ : ImpulsePositionZPlug = PlugDescriptor("impulsePositionZ")
	impulsePosition_ : ImpulsePositionPlug = PlugDescriptor("impulsePosition")
	initialOrientationX_ : InitialOrientationXPlug = PlugDescriptor("initialOrientationX")
	initialOrientationY_ : InitialOrientationYPlug = PlugDescriptor("initialOrientationY")
	initialOrientationZ_ : InitialOrientationZPlug = PlugDescriptor("initialOrientationZ")
	initialOrientation_ : InitialOrientationPlug = PlugDescriptor("initialOrientation")
	initialPositionX_ : InitialPositionXPlug = PlugDescriptor("initialPositionX")
	initialPositionY_ : InitialPositionYPlug = PlugDescriptor("initialPositionY")
	initialPositionZ_ : InitialPositionZPlug = PlugDescriptor("initialPositionZ")
	initialPosition_ : InitialPositionPlug = PlugDescriptor("initialPosition")
	initialSpinX_ : InitialSpinXPlug = PlugDescriptor("initialSpinX")
	initialSpinY_ : InitialSpinYPlug = PlugDescriptor("initialSpinY")
	initialSpinZ_ : InitialSpinZPlug = PlugDescriptor("initialSpinZ")
	initialSpin_ : InitialSpinPlug = PlugDescriptor("initialSpin")
	initialVelocityX_ : InitialVelocityXPlug = PlugDescriptor("initialVelocityX")
	initialVelocityY_ : InitialVelocityYPlug = PlugDescriptor("initialVelocityY")
	initialVelocityZ_ : InitialVelocityZPlug = PlugDescriptor("initialVelocityZ")
	initialVelocity_ : InitialVelocityPlug = PlugDescriptor("initialVelocity")
	inputForce_ : InputForcePlug = PlugDescriptor("inputForce")
	inputForceType_ : InputForceTypePlug = PlugDescriptor("inputForceType")
	inputGeometryCnt_ : InputGeometryCntPlug = PlugDescriptor("inputGeometryCnt")
	inputGeometryMsg_ : InputGeometryMsgPlug = PlugDescriptor("inputGeometryMsg")
	interpenetrateWith_ : InterpenetrateWithPlug = PlugDescriptor("interpenetrateWith")
	isKeyframed_ : IsKeyframedPlug = PlugDescriptor("isKeyframed")
	isKinematic_ : IsKinematicPlug = PlugDescriptor("isKinematic")
	isParented_ : IsParentedPlug = PlugDescriptor("isParented")
	lastCachedFrame_ : LastCachedFramePlug = PlugDescriptor("lastCachedFrame")
	lastPositionX_ : LastPositionXPlug = PlugDescriptor("lastPositionX")
	lastPositionY_ : LastPositionYPlug = PlugDescriptor("lastPositionY")
	lastPositionZ_ : LastPositionZPlug = PlugDescriptor("lastPositionZ")
	lastPosition_ : LastPositionPlug = PlugDescriptor("lastPosition")
	lastRotationX_ : LastRotationXPlug = PlugDescriptor("lastRotationX")
	lastRotationY_ : LastRotationYPlug = PlugDescriptor("lastRotationY")
	lastRotationZ_ : LastRotationZPlug = PlugDescriptor("lastRotationZ")
	lastRotation_ : LastRotationPlug = PlugDescriptor("lastRotation")
	lastSceneTime_ : LastSceneTimePlug = PlugDescriptor("lastSceneTime")
	lockCenterOfMass_ : LockCenterOfMassPlug = PlugDescriptor("lockCenterOfMass")
	mass_ : MassPlug = PlugDescriptor("mass")
	particleCollision_ : ParticleCollisionPlug = PlugDescriptor("particleCollision")
	rigidWorldMatrix_ : RigidWorldMatrixPlug = PlugDescriptor("rigidWorldMatrix")
	runUpCache_ : RunUpCachePlug = PlugDescriptor("runUpCache")
	shapeChanged_ : ShapeChangedPlug = PlugDescriptor("shapeChanged")
	solverId_ : SolverIdPlug = PlugDescriptor("solverId")
	spinX_ : SpinXPlug = PlugDescriptor("spinX")
	spinY_ : SpinYPlug = PlugDescriptor("spinY")
	spinZ_ : SpinZPlug = PlugDescriptor("spinZ")
	spin_ : SpinPlug = PlugDescriptor("spin")
	spinImpulseX_ : SpinImpulseXPlug = PlugDescriptor("spinImpulseX")
	spinImpulseY_ : SpinImpulseYPlug = PlugDescriptor("spinImpulseY")
	spinImpulseZ_ : SpinImpulseZPlug = PlugDescriptor("spinImpulseZ")
	spinImpulse_ : SpinImpulsePlug = PlugDescriptor("spinImpulse")
	standIn_ : StandInPlug = PlugDescriptor("standIn")
	staticFriction_ : StaticFrictionPlug = PlugDescriptor("staticFriction")
	tessellationFactor_ : TessellationFactorPlug = PlugDescriptor("tessellationFactor")
	torqueX_ : TorqueXPlug = PlugDescriptor("torqueX")
	torqueY_ : TorqueYPlug = PlugDescriptor("torqueY")
	torqueZ_ : TorqueZPlug = PlugDescriptor("torqueZ")
	torque_ : TorquePlug = PlugDescriptor("torque")
	velocityX_ : VelocityXPlug = PlugDescriptor("velocityX")
	velocityY_ : VelocityYPlug = PlugDescriptor("velocityY")
	velocityZ_ : VelocityZPlug = PlugDescriptor("velocityZ")
	velocity_ : VelocityPlug = PlugDescriptor("velocity")
	volume_ : VolumePlug = PlugDescriptor("volume")

	# node attributes

	typeName = "rigidBody"
	typeIdInt = 1498564420
	pass

