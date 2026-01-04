

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
class AngularVelocityXPlug(Plug):
	parent : AngularVelocityPlug = PlugDescriptor("angularVelocity")
	node : RigidConstraint = None
	pass
class AngularVelocityYPlug(Plug):
	parent : AngularVelocityPlug = PlugDescriptor("angularVelocity")
	node : RigidConstraint = None
	pass
class AngularVelocityZPlug(Plug):
	parent : AngularVelocityPlug = PlugDescriptor("angularVelocity")
	node : RigidConstraint = None
	pass
class AngularVelocityPlug(Plug):
	angularVelocityX_ : AngularVelocityXPlug = PlugDescriptor("angularVelocityX")
	avx_ : AngularVelocityXPlug = PlugDescriptor("angularVelocityX")
	angularVelocityY_ : AngularVelocityYPlug = PlugDescriptor("angularVelocityY")
	avy_ : AngularVelocityYPlug = PlugDescriptor("angularVelocityY")
	angularVelocityZ_ : AngularVelocityZPlug = PlugDescriptor("angularVelocityZ")
	avz_ : AngularVelocityZPlug = PlugDescriptor("angularVelocityZ")
	node : RigidConstraint = None
	pass
class ConstrainPlug(Plug):
	node : RigidConstraint = None
	pass
class ConstraintTypePlug(Plug):
	node : RigidConstraint = None
	pass
class ForceXPlug(Plug):
	parent : ForcePlug = PlugDescriptor("force")
	node : RigidConstraint = None
	pass
class ForceYPlug(Plug):
	parent : ForcePlug = PlugDescriptor("force")
	node : RigidConstraint = None
	pass
class ForceZPlug(Plug):
	parent : ForcePlug = PlugDescriptor("force")
	node : RigidConstraint = None
	pass
class ForcePlug(Plug):
	forceX_ : ForceXPlug = PlugDescriptor("forceX")
	frx_ : ForceXPlug = PlugDescriptor("forceX")
	forceY_ : ForceYPlug = PlugDescriptor("forceY")
	fry_ : ForceYPlug = PlugDescriptor("forceY")
	forceZ_ : ForceZPlug = PlugDescriptor("forceZ")
	frz_ : ForceZPlug = PlugDescriptor("forceZ")
	node : RigidConstraint = None
	pass
class InitialOrientationXPlug(Plug):
	parent : InitialOrientationPlug = PlugDescriptor("initialOrientation")
	node : RigidConstraint = None
	pass
class InitialOrientationYPlug(Plug):
	parent : InitialOrientationPlug = PlugDescriptor("initialOrientation")
	node : RigidConstraint = None
	pass
class InitialOrientationZPlug(Plug):
	parent : InitialOrientationPlug = PlugDescriptor("initialOrientation")
	node : RigidConstraint = None
	pass
class InitialOrientationPlug(Plug):
	initialOrientationX_ : InitialOrientationXPlug = PlugDescriptor("initialOrientationX")
	iox_ : InitialOrientationXPlug = PlugDescriptor("initialOrientationX")
	initialOrientationY_ : InitialOrientationYPlug = PlugDescriptor("initialOrientationY")
	ioy_ : InitialOrientationYPlug = PlugDescriptor("initialOrientationY")
	initialOrientationZ_ : InitialOrientationZPlug = PlugDescriptor("initialOrientationZ")
	ioz_ : InitialOrientationZPlug = PlugDescriptor("initialOrientationZ")
	node : RigidConstraint = None
	pass
class InitialPositionXPlug(Plug):
	parent : InitialPositionPlug = PlugDescriptor("initialPosition")
	node : RigidConstraint = None
	pass
class InitialPositionYPlug(Plug):
	parent : InitialPositionPlug = PlugDescriptor("initialPosition")
	node : RigidConstraint = None
	pass
class InitialPositionZPlug(Plug):
	parent : InitialPositionPlug = PlugDescriptor("initialPosition")
	node : RigidConstraint = None
	pass
class InitialPositionPlug(Plug):
	initialPositionX_ : InitialPositionXPlug = PlugDescriptor("initialPositionX")
	ipx_ : InitialPositionXPlug = PlugDescriptor("initialPositionX")
	initialPositionY_ : InitialPositionYPlug = PlugDescriptor("initialPositionY")
	ipy_ : InitialPositionYPlug = PlugDescriptor("initialPositionY")
	initialPositionZ_ : InitialPositionZPlug = PlugDescriptor("initialPositionZ")
	ipz_ : InitialPositionZPlug = PlugDescriptor("initialPositionZ")
	node : RigidConstraint = None
	pass
class InterpenetratePlug(Plug):
	node : RigidConstraint = None
	pass
class IsBoundedPlug(Plug):
	node : RigidConstraint = None
	pass
class IsParentedPlug(Plug):
	node : RigidConstraint = None
	pass
class RelativeToPlug(Plug):
	node : RigidConstraint = None
	pass
class RigidBody1Plug(Plug):
	node : RigidConstraint = None
	pass
class RigidBody2Plug(Plug):
	node : RigidConstraint = None
	pass
class SolverIdPlug(Plug):
	node : RigidConstraint = None
	pass
class SpringDampingPlug(Plug):
	node : RigidConstraint = None
	pass
class SpringRestLengthPlug(Plug):
	node : RigidConstraint = None
	pass
class SpringStiffnessPlug(Plug):
	node : RigidConstraint = None
	pass
class UserDefinedPositionXPlug(Plug):
	parent : UserDefinedPositionPlug = PlugDescriptor("userDefinedPosition")
	node : RigidConstraint = None
	pass
class UserDefinedPositionYPlug(Plug):
	parent : UserDefinedPositionPlug = PlugDescriptor("userDefinedPosition")
	node : RigidConstraint = None
	pass
class UserDefinedPositionZPlug(Plug):
	parent : UserDefinedPositionPlug = PlugDescriptor("userDefinedPosition")
	node : RigidConstraint = None
	pass
class UserDefinedPositionPlug(Plug):
	userDefinedPositionX_ : UserDefinedPositionXPlug = PlugDescriptor("userDefinedPositionX")
	upx_ : UserDefinedPositionXPlug = PlugDescriptor("userDefinedPositionX")
	userDefinedPositionY_ : UserDefinedPositionYPlug = PlugDescriptor("userDefinedPositionY")
	upy_ : UserDefinedPositionYPlug = PlugDescriptor("userDefinedPositionY")
	userDefinedPositionZ_ : UserDefinedPositionZPlug = PlugDescriptor("userDefinedPositionZ")
	upz_ : UserDefinedPositionZPlug = PlugDescriptor("userDefinedPositionZ")
	node : RigidConstraint = None
	pass
class VelocityXPlug(Plug):
	parent : VelocityPlug = PlugDescriptor("velocity")
	node : RigidConstraint = None
	pass
class VelocityYPlug(Plug):
	parent : VelocityPlug = PlugDescriptor("velocity")
	node : RigidConstraint = None
	pass
class VelocityZPlug(Plug):
	parent : VelocityPlug = PlugDescriptor("velocity")
	node : RigidConstraint = None
	pass
class VelocityPlug(Plug):
	velocityX_ : VelocityXPlug = PlugDescriptor("velocityX")
	vlx_ : VelocityXPlug = PlugDescriptor("velocityX")
	velocityY_ : VelocityYPlug = PlugDescriptor("velocityY")
	vly_ : VelocityYPlug = PlugDescriptor("velocityY")
	velocityZ_ : VelocityZPlug = PlugDescriptor("velocityZ")
	vlz_ : VelocityZPlug = PlugDescriptor("velocityZ")
	node : RigidConstraint = None
	pass
# endregion


# define node class
class RigidConstraint(Transform):
	angularVelocityX_ : AngularVelocityXPlug = PlugDescriptor("angularVelocityX")
	angularVelocityY_ : AngularVelocityYPlug = PlugDescriptor("angularVelocityY")
	angularVelocityZ_ : AngularVelocityZPlug = PlugDescriptor("angularVelocityZ")
	angularVelocity_ : AngularVelocityPlug = PlugDescriptor("angularVelocity")
	constrain_ : ConstrainPlug = PlugDescriptor("constrain")
	constraintType_ : ConstraintTypePlug = PlugDescriptor("constraintType")
	forceX_ : ForceXPlug = PlugDescriptor("forceX")
	forceY_ : ForceYPlug = PlugDescriptor("forceY")
	forceZ_ : ForceZPlug = PlugDescriptor("forceZ")
	force_ : ForcePlug = PlugDescriptor("force")
	initialOrientationX_ : InitialOrientationXPlug = PlugDescriptor("initialOrientationX")
	initialOrientationY_ : InitialOrientationYPlug = PlugDescriptor("initialOrientationY")
	initialOrientationZ_ : InitialOrientationZPlug = PlugDescriptor("initialOrientationZ")
	initialOrientation_ : InitialOrientationPlug = PlugDescriptor("initialOrientation")
	initialPositionX_ : InitialPositionXPlug = PlugDescriptor("initialPositionX")
	initialPositionY_ : InitialPositionYPlug = PlugDescriptor("initialPositionY")
	initialPositionZ_ : InitialPositionZPlug = PlugDescriptor("initialPositionZ")
	initialPosition_ : InitialPositionPlug = PlugDescriptor("initialPosition")
	interpenetrate_ : InterpenetratePlug = PlugDescriptor("interpenetrate")
	isBounded_ : IsBoundedPlug = PlugDescriptor("isBounded")
	isParented_ : IsParentedPlug = PlugDescriptor("isParented")
	relativeTo_ : RelativeToPlug = PlugDescriptor("relativeTo")
	rigidBody1_ : RigidBody1Plug = PlugDescriptor("rigidBody1")
	rigidBody2_ : RigidBody2Plug = PlugDescriptor("rigidBody2")
	solverId_ : SolverIdPlug = PlugDescriptor("solverId")
	springDamping_ : SpringDampingPlug = PlugDescriptor("springDamping")
	springRestLength_ : SpringRestLengthPlug = PlugDescriptor("springRestLength")
	springStiffness_ : SpringStiffnessPlug = PlugDescriptor("springStiffness")
	userDefinedPositionX_ : UserDefinedPositionXPlug = PlugDescriptor("userDefinedPositionX")
	userDefinedPositionY_ : UserDefinedPositionYPlug = PlugDescriptor("userDefinedPositionY")
	userDefinedPositionZ_ : UserDefinedPositionZPlug = PlugDescriptor("userDefinedPositionZ")
	userDefinedPosition_ : UserDefinedPositionPlug = PlugDescriptor("userDefinedPosition")
	velocityX_ : VelocityXPlug = PlugDescriptor("velocityX")
	velocityY_ : VelocityYPlug = PlugDescriptor("velocityY")
	velocityZ_ : VelocityZPlug = PlugDescriptor("velocityZ")
	velocity_ : VelocityPlug = PlugDescriptor("velocity")

	# node attributes

	typeName = "rigidConstraint"
	apiTypeInt = 313
	apiTypeStr = "kRigidConstraint"
	typeIdInt = 1497584468
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["angularVelocityX", "angularVelocityY", "angularVelocityZ", "angularVelocity", "constrain", "constraintType", "forceX", "forceY", "forceZ", "force", "initialOrientationX", "initialOrientationY", "initialOrientationZ", "initialOrientation", "initialPositionX", "initialPositionY", "initialPositionZ", "initialPosition", "interpenetrate", "isBounded", "isParented", "relativeTo", "rigidBody1", "rigidBody2", "solverId", "springDamping", "springRestLength", "springStiffness", "userDefinedPositionX", "userDefinedPositionY", "userDefinedPositionZ", "userDefinedPosition", "velocityX", "velocityY", "velocityZ", "velocity"]
	nodeLeafPlugs = ["angularVelocity", "constrain", "constraintType", "force", "initialOrientation", "initialPosition", "interpenetrate", "isBounded", "isParented", "relativeTo", "rigidBody1", "rigidBody2", "solverId", "springDamping", "springRestLength", "springStiffness", "userDefinedPosition", "velocity"]
	pass

