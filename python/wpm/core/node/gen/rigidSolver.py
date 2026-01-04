

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class AllowDisconnectionPlug(Plug):
	node : RigidSolver = None
	pass
class AutoSolverTolerancesPlug(Plug):
	node : RigidSolver = None
	pass
class BinMembershipPlug(Plug):
	node : RigidSolver = None
	pass
class BouncinessPlug(Plug):
	node : RigidSolver = None
	pass
class CacheDataPlug(Plug):
	node : RigidSolver = None
	pass
class CollisionTolerancePlug(Plug):
	node : RigidSolver = None
	pass
class ConstraintRotateXPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : RigidSolver = None
	pass
class ConstraintRotateYPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : RigidSolver = None
	pass
class ConstraintRotateZPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : RigidSolver = None
	pass
class ConstraintRotatePlug(Plug):
	constraintRotateX_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	crx_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	constraintRotateY_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	cry_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	constraintRotateZ_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	crz_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	node : RigidSolver = None
	pass
class ConstraintTranslateXPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : RigidSolver = None
	pass
class ConstraintTranslateYPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : RigidSolver = None
	pass
class ConstraintTranslateZPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : RigidSolver = None
	pass
class ConstraintTranslatePlug(Plug):
	constraintTranslateX_ : ConstraintTranslateXPlug = PlugDescriptor("constraintTranslateX")
	ctx_ : ConstraintTranslateXPlug = PlugDescriptor("constraintTranslateX")
	constraintTranslateY_ : ConstraintTranslateYPlug = PlugDescriptor("constraintTranslateY")
	cty_ : ConstraintTranslateYPlug = PlugDescriptor("constraintTranslateY")
	constraintTranslateZ_ : ConstraintTranslateZPlug = PlugDescriptor("constraintTranslateZ")
	ctz_ : ConstraintTranslateZPlug = PlugDescriptor("constraintTranslateZ")
	node : RigidSolver = None
	pass
class ContactDataPlug(Plug):
	node : RigidSolver = None
	pass
class CurrentPlug(Plug):
	node : RigidSolver = None
	pass
class CurrentTimePlug(Plug):
	node : RigidSolver = None
	pass
class DeltaTimePlug(Plug):
	node : RigidSolver = None
	pass
class DisplayCenterOfMassPlug(Plug):
	node : RigidSolver = None
	pass
class DisplayConstraintPlug(Plug):
	node : RigidSolver = None
	pass
class DisplayLabelPlug(Plug):
	node : RigidSolver = None
	pass
class DisplayVelocityPlug(Plug):
	node : RigidSolver = None
	pass
class DynamicsPlug(Plug):
	node : RigidSolver = None
	pass
class ForceDynamicsPlug(Plug):
	node : RigidSolver = None
	pass
class FrictionPlug(Plug):
	node : RigidSolver = None
	pass
class InputForcePlug(Plug):
	parent : GeneralForcePlug = PlugDescriptor("generalForce")
	node : RigidSolver = None
	pass
class InputTorquePlug(Plug):
	parent : GeneralForcePlug = PlugDescriptor("generalForce")
	node : RigidSolver = None
	pass
class GeneralForcePlug(Plug):
	inputForce_ : InputForcePlug = PlugDescriptor("inputForce")
	ifr_ : InputForcePlug = PlugDescriptor("inputForce")
	inputTorque_ : InputTorquePlug = PlugDescriptor("inputTorque")
	itr_ : InputTorquePlug = PlugDescriptor("inputTorque")
	node : RigidSolver = None
	pass
class LastSceneTimePlug(Plug):
	node : RigidSolver = None
	pass
class RigidBodyCountPlug(Plug):
	node : RigidSolver = None
	pass
class RotateXPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : RigidSolver = None
	pass
class RotateYPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : RigidSolver = None
	pass
class RotateZPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : RigidSolver = None
	pass
class RotatePlug(Plug):
	rotateX_ : RotateXPlug = PlugDescriptor("rotateX")
	rx_ : RotateXPlug = PlugDescriptor("rotateX")
	rotateY_ : RotateYPlug = PlugDescriptor("rotateY")
	ry_ : RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_ : RotateZPlug = PlugDescriptor("rotateZ")
	rz_ : RotateZPlug = PlugDescriptor("rotateZ")
	node : RigidSolver = None
	pass
class ScaleVelocityPlug(Plug):
	node : RigidSolver = None
	pass
class SolverMethodPlug(Plug):
	node : RigidSolver = None
	pass
class SolvingPlug(Plug):
	node : RigidSolver = None
	pass
class StartTimePlug(Plug):
	node : RigidSolver = None
	pass
class StatePlug(Plug):
	node : RigidSolver = None
	pass
class StatisticsPlug(Plug):
	node : RigidSolver = None
	pass
class StepSizePlug(Plug):
	node : RigidSolver = None
	pass
class TranslateXPlug(Plug):
	parent : TranslatePlug = PlugDescriptor("translate")
	node : RigidSolver = None
	pass
class TranslateYPlug(Plug):
	parent : TranslatePlug = PlugDescriptor("translate")
	node : RigidSolver = None
	pass
class TranslateZPlug(Plug):
	parent : TranslatePlug = PlugDescriptor("translate")
	node : RigidSolver = None
	pass
class TranslatePlug(Plug):
	translateX_ : TranslateXPlug = PlugDescriptor("translateX")
	tx_ : TranslateXPlug = PlugDescriptor("translateX")
	translateY_ : TranslateYPlug = PlugDescriptor("translateY")
	ty_ : TranslateYPlug = PlugDescriptor("translateY")
	translateZ_ : TranslateZPlug = PlugDescriptor("translateZ")
	tz_ : TranslateZPlug = PlugDescriptor("translateZ")
	node : RigidSolver = None
	pass
# endregion


# define node class
class RigidSolver(_BASE_):
	allowDisconnection_ : AllowDisconnectionPlug = PlugDescriptor("allowDisconnection")
	autoSolverTolerances_ : AutoSolverTolerancesPlug = PlugDescriptor("autoSolverTolerances")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	bounciness_ : BouncinessPlug = PlugDescriptor("bounciness")
	cacheData_ : CacheDataPlug = PlugDescriptor("cacheData")
	collisionTolerance_ : CollisionTolerancePlug = PlugDescriptor("collisionTolerance")
	constraintRotateX_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	constraintRotateY_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	constraintRotateZ_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	constraintRotate_ : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	constraintTranslateX_ : ConstraintTranslateXPlug = PlugDescriptor("constraintTranslateX")
	constraintTranslateY_ : ConstraintTranslateYPlug = PlugDescriptor("constraintTranslateY")
	constraintTranslateZ_ : ConstraintTranslateZPlug = PlugDescriptor("constraintTranslateZ")
	constraintTranslate_ : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	contactData_ : ContactDataPlug = PlugDescriptor("contactData")
	current_ : CurrentPlug = PlugDescriptor("current")
	currentTime_ : CurrentTimePlug = PlugDescriptor("currentTime")
	deltaTime_ : DeltaTimePlug = PlugDescriptor("deltaTime")
	displayCenterOfMass_ : DisplayCenterOfMassPlug = PlugDescriptor("displayCenterOfMass")
	displayConstraint_ : DisplayConstraintPlug = PlugDescriptor("displayConstraint")
	displayLabel_ : DisplayLabelPlug = PlugDescriptor("displayLabel")
	displayVelocity_ : DisplayVelocityPlug = PlugDescriptor("displayVelocity")
	dynamics_ : DynamicsPlug = PlugDescriptor("dynamics")
	forceDynamics_ : ForceDynamicsPlug = PlugDescriptor("forceDynamics")
	friction_ : FrictionPlug = PlugDescriptor("friction")
	inputForce_ : InputForcePlug = PlugDescriptor("inputForce")
	inputTorque_ : InputTorquePlug = PlugDescriptor("inputTorque")
	generalForce_ : GeneralForcePlug = PlugDescriptor("generalForce")
	lastSceneTime_ : LastSceneTimePlug = PlugDescriptor("lastSceneTime")
	rigidBodyCount_ : RigidBodyCountPlug = PlugDescriptor("rigidBodyCount")
	rotateX_ : RotateXPlug = PlugDescriptor("rotateX")
	rotateY_ : RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_ : RotateZPlug = PlugDescriptor("rotateZ")
	rotate_ : RotatePlug = PlugDescriptor("rotate")
	scaleVelocity_ : ScaleVelocityPlug = PlugDescriptor("scaleVelocity")
	solverMethod_ : SolverMethodPlug = PlugDescriptor("solverMethod")
	solving_ : SolvingPlug = PlugDescriptor("solving")
	startTime_ : StartTimePlug = PlugDescriptor("startTime")
	state_ : StatePlug = PlugDescriptor("state")
	statistics_ : StatisticsPlug = PlugDescriptor("statistics")
	stepSize_ : StepSizePlug = PlugDescriptor("stepSize")
	translateX_ : TranslateXPlug = PlugDescriptor("translateX")
	translateY_ : TranslateYPlug = PlugDescriptor("translateY")
	translateZ_ : TranslateZPlug = PlugDescriptor("translateZ")
	translate_ : TranslatePlug = PlugDescriptor("translate")

	# node attributes

	typeName = "rigidSolver"
	apiTypeInt = 470
	apiTypeStr = "kRigidSolver"
	typeIdInt = 1498631254
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["allowDisconnection", "autoSolverTolerances", "binMembership", "bounciness", "cacheData", "collisionTolerance", "constraintRotateX", "constraintRotateY", "constraintRotateZ", "constraintRotate", "constraintTranslateX", "constraintTranslateY", "constraintTranslateZ", "constraintTranslate", "contactData", "current", "currentTime", "deltaTime", "displayCenterOfMass", "displayConstraint", "displayLabel", "displayVelocity", "dynamics", "forceDynamics", "friction", "inputForce", "inputTorque", "generalForce", "lastSceneTime", "rigidBodyCount", "rotateX", "rotateY", "rotateZ", "rotate", "scaleVelocity", "solverMethod", "solving", "startTime", "state", "statistics", "stepSize", "translateX", "translateY", "translateZ", "translate"]
	nodeLeafPlugs = ["allowDisconnection", "autoSolverTolerances", "binMembership", "bounciness", "cacheData", "collisionTolerance", "constraintRotate", "constraintTranslate", "contactData", "current", "currentTime", "deltaTime", "displayCenterOfMass", "displayConstraint", "displayLabel", "displayVelocity", "dynamics", "forceDynamics", "friction", "generalForce", "lastSceneTime", "rigidBodyCount", "rotate", "scaleVelocity", "solverMethod", "solving", "startTime", "state", "statistics", "stepSize", "translate"]
	pass

