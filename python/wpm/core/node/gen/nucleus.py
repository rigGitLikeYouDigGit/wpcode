

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Transform = Catalogue.Transform
else:
	from .. import retriever
	Transform = retriever.getNodeCls("Transform")
	assert Transform

# add node doc



# region plug type defs
class AirDensityPlug(Plug):
	node : Nucleus = None
	pass
class CollisionFlagPlug(Plug):
	node : Nucleus = None
	pass
class CollisionLayerRangePlug(Plug):
	node : Nucleus = None
	pass
class CollisionSoftnessPlug(Plug):
	node : Nucleus = None
	pass
class CurrentTimePlug(Plug):
	node : Nucleus = None
	pass
class EnablePlug(Plug):
	node : Nucleus = None
	pass
class EvalIdPlug(Plug):
	node : Nucleus = None
	pass
class ForceDynamicsPlug(Plug):
	node : Nucleus = None
	pass
class FrameJumpLimitPlug(Plug):
	node : Nucleus = None
	pass
class GravityPlug(Plug):
	node : Nucleus = None
	pass
class GravityDirectionXPlug(Plug):
	parent : GravityDirectionPlug = PlugDescriptor("gravityDirection")
	node : Nucleus = None
	pass
class GravityDirectionYPlug(Plug):
	parent : GravityDirectionPlug = PlugDescriptor("gravityDirection")
	node : Nucleus = None
	pass
class GravityDirectionZPlug(Plug):
	parent : GravityDirectionPlug = PlugDescriptor("gravityDirection")
	node : Nucleus = None
	pass
class GravityDirectionPlug(Plug):
	gravityDirectionX_ : GravityDirectionXPlug = PlugDescriptor("gravityDirectionX")
	grdx_ : GravityDirectionXPlug = PlugDescriptor("gravityDirectionX")
	gravityDirectionY_ : GravityDirectionYPlug = PlugDescriptor("gravityDirectionY")
	grdy_ : GravityDirectionYPlug = PlugDescriptor("gravityDirectionY")
	gravityDirectionZ_ : GravityDirectionZPlug = PlugDescriptor("gravityDirectionZ")
	grdz_ : GravityDirectionZPlug = PlugDescriptor("gravityDirectionZ")
	node : Nucleus = None
	pass
class InputActivePlug(Plug):
	node : Nucleus = None
	pass
class InputActiveStartPlug(Plug):
	node : Nucleus = None
	pass
class InputCurrentPlug(Plug):
	node : Nucleus = None
	pass
class InputPassivePlug(Plug):
	node : Nucleus = None
	pass
class InputPassiveStartPlug(Plug):
	node : Nucleus = None
	pass
class InputStartPlug(Plug):
	node : Nucleus = None
	pass
class LastTimePlug(Plug):
	node : Nucleus = None
	pass
class MaxCollisionIterationsPlug(Plug):
	node : Nucleus = None
	pass
class OutputObjectsPlug(Plug):
	node : Nucleus = None
	pass
class PlaneBouncePlug(Plug):
	node : Nucleus = None
	pass
class PlaneFrictionPlug(Plug):
	node : Nucleus = None
	pass
class PlaneNormalXPlug(Plug):
	parent : PlaneNormalPlug = PlugDescriptor("planeNormal")
	node : Nucleus = None
	pass
class PlaneNormalYPlug(Plug):
	parent : PlaneNormalPlug = PlugDescriptor("planeNormal")
	node : Nucleus = None
	pass
class PlaneNormalZPlug(Plug):
	parent : PlaneNormalPlug = PlugDescriptor("planeNormal")
	node : Nucleus = None
	pass
class PlaneNormalPlug(Plug):
	planeNormalX_ : PlaneNormalXPlug = PlugDescriptor("planeNormalX")
	npnx_ : PlaneNormalXPlug = PlugDescriptor("planeNormalX")
	planeNormalY_ : PlaneNormalYPlug = PlugDescriptor("planeNormalY")
	npny_ : PlaneNormalYPlug = PlugDescriptor("planeNormalY")
	planeNormalZ_ : PlaneNormalZPlug = PlugDescriptor("planeNormalZ")
	npnz_ : PlaneNormalZPlug = PlugDescriptor("planeNormalZ")
	node : Nucleus = None
	pass
class PlaneOriginXPlug(Plug):
	parent : PlaneOriginPlug = PlugDescriptor("planeOrigin")
	node : Nucleus = None
	pass
class PlaneOriginYPlug(Plug):
	parent : PlaneOriginPlug = PlugDescriptor("planeOrigin")
	node : Nucleus = None
	pass
class PlaneOriginZPlug(Plug):
	parent : PlaneOriginPlug = PlugDescriptor("planeOrigin")
	node : Nucleus = None
	pass
class PlaneOriginPlug(Plug):
	planeOriginX_ : PlaneOriginXPlug = PlugDescriptor("planeOriginX")
	npox_ : PlaneOriginXPlug = PlugDescriptor("planeOriginX")
	planeOriginY_ : PlaneOriginYPlug = PlugDescriptor("planeOriginY")
	npoy_ : PlaneOriginYPlug = PlugDescriptor("planeOriginY")
	planeOriginZ_ : PlaneOriginZPlug = PlugDescriptor("planeOriginZ")
	npoz_ : PlaneOriginZPlug = PlugDescriptor("planeOriginZ")
	node : Nucleus = None
	pass
class PlaneStickinessPlug(Plug):
	node : Nucleus = None
	pass
class SelfCollisionFlagPlug(Plug):
	node : Nucleus = None
	pass
class SkipSetupPlug(Plug):
	node : Nucleus = None
	pass
class SpaceScalePlug(Plug):
	node : Nucleus = None
	pass
class StartFramePlug(Plug):
	node : Nucleus = None
	pass
class StartTimePlug(Plug):
	node : Nucleus = None
	pass
class SubStepsPlug(Plug):
	node : Nucleus = None
	pass
class TimeScalePlug(Plug):
	node : Nucleus = None
	pass
class TimingOutputPlug(Plug):
	node : Nucleus = None
	pass
class UsePlanePlug(Plug):
	node : Nucleus = None
	pass
class UseTransformPlug(Plug):
	node : Nucleus = None
	pass
class WindDirectionXPlug(Plug):
	parent : WindDirectionPlug = PlugDescriptor("windDirection")
	node : Nucleus = None
	pass
class WindDirectionYPlug(Plug):
	parent : WindDirectionPlug = PlugDescriptor("windDirection")
	node : Nucleus = None
	pass
class WindDirectionZPlug(Plug):
	parent : WindDirectionPlug = PlugDescriptor("windDirection")
	node : Nucleus = None
	pass
class WindDirectionPlug(Plug):
	windDirectionX_ : WindDirectionXPlug = PlugDescriptor("windDirectionX")
	widx_ : WindDirectionXPlug = PlugDescriptor("windDirectionX")
	windDirectionY_ : WindDirectionYPlug = PlugDescriptor("windDirectionY")
	widy_ : WindDirectionYPlug = PlugDescriptor("windDirectionY")
	windDirectionZ_ : WindDirectionZPlug = PlugDescriptor("windDirectionZ")
	widz_ : WindDirectionZPlug = PlugDescriptor("windDirectionZ")
	node : Nucleus = None
	pass
class WindNoisePlug(Plug):
	node : Nucleus = None
	pass
class WindSpeedPlug(Plug):
	node : Nucleus = None
	pass
# endregion


# define node class
class Nucleus(Transform):
	airDensity_ : AirDensityPlug = PlugDescriptor("airDensity")
	collisionFlag_ : CollisionFlagPlug = PlugDescriptor("collisionFlag")
	collisionLayerRange_ : CollisionLayerRangePlug = PlugDescriptor("collisionLayerRange")
	collisionSoftness_ : CollisionSoftnessPlug = PlugDescriptor("collisionSoftness")
	currentTime_ : CurrentTimePlug = PlugDescriptor("currentTime")
	enable_ : EnablePlug = PlugDescriptor("enable")
	evalId_ : EvalIdPlug = PlugDescriptor("evalId")
	forceDynamics_ : ForceDynamicsPlug = PlugDescriptor("forceDynamics")
	frameJumpLimit_ : FrameJumpLimitPlug = PlugDescriptor("frameJumpLimit")
	gravity_ : GravityPlug = PlugDescriptor("gravity")
	gravityDirectionX_ : GravityDirectionXPlug = PlugDescriptor("gravityDirectionX")
	gravityDirectionY_ : GravityDirectionYPlug = PlugDescriptor("gravityDirectionY")
	gravityDirectionZ_ : GravityDirectionZPlug = PlugDescriptor("gravityDirectionZ")
	gravityDirection_ : GravityDirectionPlug = PlugDescriptor("gravityDirection")
	inputActive_ : InputActivePlug = PlugDescriptor("inputActive")
	inputActiveStart_ : InputActiveStartPlug = PlugDescriptor("inputActiveStart")
	inputCurrent_ : InputCurrentPlug = PlugDescriptor("inputCurrent")
	inputPassive_ : InputPassivePlug = PlugDescriptor("inputPassive")
	inputPassiveStart_ : InputPassiveStartPlug = PlugDescriptor("inputPassiveStart")
	inputStart_ : InputStartPlug = PlugDescriptor("inputStart")
	lastTime_ : LastTimePlug = PlugDescriptor("lastTime")
	maxCollisionIterations_ : MaxCollisionIterationsPlug = PlugDescriptor("maxCollisionIterations")
	outputObjects_ : OutputObjectsPlug = PlugDescriptor("outputObjects")
	planeBounce_ : PlaneBouncePlug = PlugDescriptor("planeBounce")
	planeFriction_ : PlaneFrictionPlug = PlugDescriptor("planeFriction")
	planeNormalX_ : PlaneNormalXPlug = PlugDescriptor("planeNormalX")
	planeNormalY_ : PlaneNormalYPlug = PlugDescriptor("planeNormalY")
	planeNormalZ_ : PlaneNormalZPlug = PlugDescriptor("planeNormalZ")
	planeNormal_ : PlaneNormalPlug = PlugDescriptor("planeNormal")
	planeOriginX_ : PlaneOriginXPlug = PlugDescriptor("planeOriginX")
	planeOriginY_ : PlaneOriginYPlug = PlugDescriptor("planeOriginY")
	planeOriginZ_ : PlaneOriginZPlug = PlugDescriptor("planeOriginZ")
	planeOrigin_ : PlaneOriginPlug = PlugDescriptor("planeOrigin")
	planeStickiness_ : PlaneStickinessPlug = PlugDescriptor("planeStickiness")
	selfCollisionFlag_ : SelfCollisionFlagPlug = PlugDescriptor("selfCollisionFlag")
	skipSetup_ : SkipSetupPlug = PlugDescriptor("skipSetup")
	spaceScale_ : SpaceScalePlug = PlugDescriptor("spaceScale")
	startFrame_ : StartFramePlug = PlugDescriptor("startFrame")
	startTime_ : StartTimePlug = PlugDescriptor("startTime")
	subSteps_ : SubStepsPlug = PlugDescriptor("subSteps")
	timeScale_ : TimeScalePlug = PlugDescriptor("timeScale")
	timingOutput_ : TimingOutputPlug = PlugDescriptor("timingOutput")
	usePlane_ : UsePlanePlug = PlugDescriptor("usePlane")
	useTransform_ : UseTransformPlug = PlugDescriptor("useTransform")
	windDirectionX_ : WindDirectionXPlug = PlugDescriptor("windDirectionX")
	windDirectionY_ : WindDirectionYPlug = PlugDescriptor("windDirectionY")
	windDirectionZ_ : WindDirectionZPlug = PlugDescriptor("windDirectionZ")
	windDirection_ : WindDirectionPlug = PlugDescriptor("windDirection")
	windNoise_ : WindNoisePlug = PlugDescriptor("windNoise")
	windSpeed_ : WindSpeedPlug = PlugDescriptor("windSpeed")

	# node attributes

	typeName = "nucleus"
	apiTypeInt = 997
	apiTypeStr = "kNucleus"
	typeIdInt = 1314085203
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["airDensity", "collisionFlag", "collisionLayerRange", "collisionSoftness", "currentTime", "enable", "evalId", "forceDynamics", "frameJumpLimit", "gravity", "gravityDirectionX", "gravityDirectionY", "gravityDirectionZ", "gravityDirection", "inputActive", "inputActiveStart", "inputCurrent", "inputPassive", "inputPassiveStart", "inputStart", "lastTime", "maxCollisionIterations", "outputObjects", "planeBounce", "planeFriction", "planeNormalX", "planeNormalY", "planeNormalZ", "planeNormal", "planeOriginX", "planeOriginY", "planeOriginZ", "planeOrigin", "planeStickiness", "selfCollisionFlag", "skipSetup", "spaceScale", "startFrame", "startTime", "subSteps", "timeScale", "timingOutput", "usePlane", "useTransform", "windDirectionX", "windDirectionY", "windDirectionZ", "windDirection", "windNoise", "windSpeed"]
	nodeLeafPlugs = ["airDensity", "collisionFlag", "collisionLayerRange", "collisionSoftness", "currentTime", "enable", "evalId", "forceDynamics", "frameJumpLimit", "gravity", "gravityDirection", "inputActive", "inputActiveStart", "inputCurrent", "inputPassive", "inputPassiveStart", "inputStart", "lastTime", "maxCollisionIterations", "outputObjects", "planeBounce", "planeFriction", "planeNormal", "planeOrigin", "planeStickiness", "selfCollisionFlag", "skipSetup", "spaceScale", "startFrame", "startTime", "subSteps", "timeScale", "timingOutput", "usePlane", "useTransform", "windDirection", "windNoise", "windSpeed"]
	pass

