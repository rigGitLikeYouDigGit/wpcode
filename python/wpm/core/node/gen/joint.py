

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
class BindInverseScaleXPlug(Plug):
	parent : BindInverseScalePlug = PlugDescriptor("bindInverseScale")
	node : Joint = None
	pass
class BindInverseScaleYPlug(Plug):
	parent : BindInverseScalePlug = PlugDescriptor("bindInverseScale")
	node : Joint = None
	pass
class BindInverseScaleZPlug(Plug):
	parent : BindInverseScalePlug = PlugDescriptor("bindInverseScale")
	node : Joint = None
	pass
class BindInverseScalePlug(Plug):
	bindInverseScaleX_ : BindInverseScaleXPlug = PlugDescriptor("bindInverseScaleX")
	bix_ : BindInverseScaleXPlug = PlugDescriptor("bindInverseScaleX")
	bindInverseScaleY_ : BindInverseScaleYPlug = PlugDescriptor("bindInverseScaleY")
	biy_ : BindInverseScaleYPlug = PlugDescriptor("bindInverseScaleY")
	bindInverseScaleZ_ : BindInverseScaleZPlug = PlugDescriptor("bindInverseScaleZ")
	biz_ : BindInverseScaleZPlug = PlugDescriptor("bindInverseScaleZ")
	node : Joint = None
	pass
class BindJointOrientXPlug(Plug):
	parent : BindJointOrientPlug = PlugDescriptor("bindJointOrient")
	node : Joint = None
	pass
class BindJointOrientYPlug(Plug):
	parent : BindJointOrientPlug = PlugDescriptor("bindJointOrient")
	node : Joint = None
	pass
class BindJointOrientZPlug(Plug):
	parent : BindJointOrientPlug = PlugDescriptor("bindJointOrient")
	node : Joint = None
	pass
class BindJointOrientPlug(Plug):
	bindJointOrientX_ : BindJointOrientXPlug = PlugDescriptor("bindJointOrientX")
	bjx_ : BindJointOrientXPlug = PlugDescriptor("bindJointOrientX")
	bindJointOrientY_ : BindJointOrientYPlug = PlugDescriptor("bindJointOrientY")
	bjy_ : BindJointOrientYPlug = PlugDescriptor("bindJointOrientY")
	bindJointOrientZ_ : BindJointOrientZPlug = PlugDescriptor("bindJointOrientZ")
	bjz_ : BindJointOrientZPlug = PlugDescriptor("bindJointOrientZ")
	node : Joint = None
	pass
class BindPosePlug(Plug):
	node : Joint = None
	pass
class BindRotateAxisXPlug(Plug):
	parent : BindRotateAxisPlug = PlugDescriptor("bindRotateAxis")
	node : Joint = None
	pass
class BindRotateAxisYPlug(Plug):
	parent : BindRotateAxisPlug = PlugDescriptor("bindRotateAxis")
	node : Joint = None
	pass
class BindRotateAxisZPlug(Plug):
	parent : BindRotateAxisPlug = PlugDescriptor("bindRotateAxis")
	node : Joint = None
	pass
class BindRotateAxisPlug(Plug):
	bindRotateAxisX_ : BindRotateAxisXPlug = PlugDescriptor("bindRotateAxisX")
	brax_ : BindRotateAxisXPlug = PlugDescriptor("bindRotateAxisX")
	bindRotateAxisY_ : BindRotateAxisYPlug = PlugDescriptor("bindRotateAxisY")
	bray_ : BindRotateAxisYPlug = PlugDescriptor("bindRotateAxisY")
	bindRotateAxisZ_ : BindRotateAxisZPlug = PlugDescriptor("bindRotateAxisZ")
	braz_ : BindRotateAxisZPlug = PlugDescriptor("bindRotateAxisZ")
	node : Joint = None
	pass
class BindRotationXPlug(Plug):
	parent : BindRotationPlug = PlugDescriptor("bindRotation")
	node : Joint = None
	pass
class BindRotationYPlug(Plug):
	parent : BindRotationPlug = PlugDescriptor("bindRotation")
	node : Joint = None
	pass
class BindRotationZPlug(Plug):
	parent : BindRotationPlug = PlugDescriptor("bindRotation")
	node : Joint = None
	pass
class BindRotationPlug(Plug):
	bindRotationX_ : BindRotationXPlug = PlugDescriptor("bindRotationX")
	brx_ : BindRotationXPlug = PlugDescriptor("bindRotationX")
	bindRotationY_ : BindRotationYPlug = PlugDescriptor("bindRotationY")
	bry_ : BindRotationYPlug = PlugDescriptor("bindRotationY")
	bindRotationZ_ : BindRotationZPlug = PlugDescriptor("bindRotationZ")
	brz_ : BindRotationZPlug = PlugDescriptor("bindRotationZ")
	node : Joint = None
	pass
class BindScaleXPlug(Plug):
	parent : BindScalePlug = PlugDescriptor("bindScale")
	node : Joint = None
	pass
class BindScaleYPlug(Plug):
	parent : BindScalePlug = PlugDescriptor("bindScale")
	node : Joint = None
	pass
class BindScaleZPlug(Plug):
	parent : BindScalePlug = PlugDescriptor("bindScale")
	node : Joint = None
	pass
class BindScalePlug(Plug):
	bindScaleX_ : BindScaleXPlug = PlugDescriptor("bindScaleX")
	bsx_ : BindScaleXPlug = PlugDescriptor("bindScaleX")
	bindScaleY_ : BindScaleYPlug = PlugDescriptor("bindScaleY")
	bsy_ : BindScaleYPlug = PlugDescriptor("bindScaleY")
	bindScaleZ_ : BindScaleZPlug = PlugDescriptor("bindScaleZ")
	bsz_ : BindScaleZPlug = PlugDescriptor("bindScaleZ")
	node : Joint = None
	pass
class BindSegmentScaleCompensatePlug(Plug):
	node : Joint = None
	pass
class DofMaskPlug(Plug):
	node : Joint = None
	pass
class DrawLabelPlug(Plug):
	node : Joint = None
	pass
class DrawStylePlug(Plug):
	node : Joint = None
	pass
class FkRotateXPlug(Plug):
	parent : FkRotatePlug = PlugDescriptor("fkRotate")
	node : Joint = None
	pass
class FkRotateYPlug(Plug):
	parent : FkRotatePlug = PlugDescriptor("fkRotate")
	node : Joint = None
	pass
class FkRotateZPlug(Plug):
	parent : FkRotatePlug = PlugDescriptor("fkRotate")
	node : Joint = None
	pass
class FkRotatePlug(Plug):
	fkRotateX_ : FkRotateXPlug = PlugDescriptor("fkRotateX")
	frx_ : FkRotateXPlug = PlugDescriptor("fkRotateX")
	fkRotateY_ : FkRotateYPlug = PlugDescriptor("fkRotateY")
	fry_ : FkRotateYPlug = PlugDescriptor("fkRotateY")
	fkRotateZ_ : FkRotateZPlug = PlugDescriptor("fkRotateZ")
	frz_ : FkRotateZPlug = PlugDescriptor("fkRotateZ")
	node : Joint = None
	pass
class HikFkJointPlug(Plug):
	node : Joint = None
	pass
class HikNodeIDPlug(Plug):
	node : Joint = None
	pass
class IkRotateXPlug(Plug):
	parent : IkRotatePlug = PlugDescriptor("ikRotate")
	node : Joint = None
	pass
class IkRotateYPlug(Plug):
	parent : IkRotatePlug = PlugDescriptor("ikRotate")
	node : Joint = None
	pass
class IkRotateZPlug(Plug):
	parent : IkRotatePlug = PlugDescriptor("ikRotate")
	node : Joint = None
	pass
class IkRotatePlug(Plug):
	ikRotateX_ : IkRotateXPlug = PlugDescriptor("ikRotateX")
	irx_ : IkRotateXPlug = PlugDescriptor("ikRotateX")
	ikRotateY_ : IkRotateYPlug = PlugDescriptor("ikRotateY")
	iry_ : IkRotateYPlug = PlugDescriptor("ikRotateY")
	ikRotateZ_ : IkRotateZPlug = PlugDescriptor("ikRotateZ")
	irz_ : IkRotateZPlug = PlugDescriptor("ikRotateZ")
	node : Joint = None
	pass
class InIKSolveFlagPlug(Plug):
	node : Joint = None
	pass
class InverseScaleXPlug(Plug):
	parent : InverseScalePlug = PlugDescriptor("inverseScale")
	node : Joint = None
	pass
class InverseScaleYPlug(Plug):
	parent : InverseScalePlug = PlugDescriptor("inverseScale")
	node : Joint = None
	pass
class InverseScaleZPlug(Plug):
	parent : InverseScalePlug = PlugDescriptor("inverseScale")
	node : Joint = None
	pass
class InverseScalePlug(Plug):
	inverseScaleX_ : InverseScaleXPlug = PlugDescriptor("inverseScaleX")
	isx_ : InverseScaleXPlug = PlugDescriptor("inverseScaleX")
	inverseScaleY_ : InverseScaleYPlug = PlugDescriptor("inverseScaleY")
	isy_ : InverseScaleYPlug = PlugDescriptor("inverseScaleY")
	inverseScaleZ_ : InverseScaleZPlug = PlugDescriptor("inverseScaleZ")
	isz_ : InverseScaleZPlug = PlugDescriptor("inverseScaleZ")
	node : Joint = None
	pass
class IsIKDirtyFlagPlug(Plug):
	node : Joint = None
	pass
class JointOrientXPlug(Plug):
	parent : JointOrientPlug = PlugDescriptor("jointOrient")
	node : Joint = None
	pass
class JointOrientYPlug(Plug):
	parent : JointOrientPlug = PlugDescriptor("jointOrient")
	node : Joint = None
	pass
class JointOrientZPlug(Plug):
	parent : JointOrientPlug = PlugDescriptor("jointOrient")
	node : Joint = None
	pass
class JointOrientPlug(Plug):
	jointOrientX_ : JointOrientXPlug = PlugDescriptor("jointOrientX")
	jox_ : JointOrientXPlug = PlugDescriptor("jointOrientX")
	jointOrientY_ : JointOrientYPlug = PlugDescriptor("jointOrientY")
	joy_ : JointOrientYPlug = PlugDescriptor("jointOrientY")
	jointOrientZ_ : JointOrientZPlug = PlugDescriptor("jointOrientZ")
	joz_ : JointOrientZPlug = PlugDescriptor("jointOrientZ")
	node : Joint = None
	pass
class JointOrientTypePlug(Plug):
	node : Joint = None
	pass
class JointTypePlug(Plug):
	node : Joint = None
	pass
class JointTypeXPlug(Plug):
	node : Joint = None
	pass
class JointTypeYPlug(Plug):
	node : Joint = None
	pass
class JointTypeZPlug(Plug):
	node : Joint = None
	pass
class MaxRotateDampRangeXPlug(Plug):
	parent : MaxRotateDampRangePlug = PlugDescriptor("maxRotateDampRange")
	node : Joint = None
	pass
class MaxRotateDampRangeYPlug(Plug):
	parent : MaxRotateDampRangePlug = PlugDescriptor("maxRotateDampRange")
	node : Joint = None
	pass
class MaxRotateDampRangeZPlug(Plug):
	parent : MaxRotateDampRangePlug = PlugDescriptor("maxRotateDampRange")
	node : Joint = None
	pass
class MaxRotateDampRangePlug(Plug):
	maxRotateDampRangeX_ : MaxRotateDampRangeXPlug = PlugDescriptor("maxRotateDampRangeX")
	xdx_ : MaxRotateDampRangeXPlug = PlugDescriptor("maxRotateDampRangeX")
	maxRotateDampRangeY_ : MaxRotateDampRangeYPlug = PlugDescriptor("maxRotateDampRangeY")
	xdy_ : MaxRotateDampRangeYPlug = PlugDescriptor("maxRotateDampRangeY")
	maxRotateDampRangeZ_ : MaxRotateDampRangeZPlug = PlugDescriptor("maxRotateDampRangeZ")
	xdz_ : MaxRotateDampRangeZPlug = PlugDescriptor("maxRotateDampRangeZ")
	node : Joint = None
	pass
class MaxRotateDampStrengthXPlug(Plug):
	parent : MaxRotateDampStrengthPlug = PlugDescriptor("maxRotateDampStrength")
	node : Joint = None
	pass
class MaxRotateDampStrengthYPlug(Plug):
	parent : MaxRotateDampStrengthPlug = PlugDescriptor("maxRotateDampStrength")
	node : Joint = None
	pass
class MaxRotateDampStrengthZPlug(Plug):
	parent : MaxRotateDampStrengthPlug = PlugDescriptor("maxRotateDampStrength")
	node : Joint = None
	pass
class MaxRotateDampStrengthPlug(Plug):
	maxRotateDampStrengthX_ : MaxRotateDampStrengthXPlug = PlugDescriptor("maxRotateDampStrengthX")
	xstx_ : MaxRotateDampStrengthXPlug = PlugDescriptor("maxRotateDampStrengthX")
	maxRotateDampStrengthY_ : MaxRotateDampStrengthYPlug = PlugDescriptor("maxRotateDampStrengthY")
	xsty_ : MaxRotateDampStrengthYPlug = PlugDescriptor("maxRotateDampStrengthY")
	maxRotateDampStrengthZ_ : MaxRotateDampStrengthZPlug = PlugDescriptor("maxRotateDampStrengthZ")
	xstz_ : MaxRotateDampStrengthZPlug = PlugDescriptor("maxRotateDampStrengthZ")
	node : Joint = None
	pass
class MinRotateDampRangeXPlug(Plug):
	parent : MinRotateDampRangePlug = PlugDescriptor("minRotateDampRange")
	node : Joint = None
	pass
class MinRotateDampRangeYPlug(Plug):
	parent : MinRotateDampRangePlug = PlugDescriptor("minRotateDampRange")
	node : Joint = None
	pass
class MinRotateDampRangeZPlug(Plug):
	parent : MinRotateDampRangePlug = PlugDescriptor("minRotateDampRange")
	node : Joint = None
	pass
class MinRotateDampRangePlug(Plug):
	minRotateDampRangeX_ : MinRotateDampRangeXPlug = PlugDescriptor("minRotateDampRangeX")
	ndx_ : MinRotateDampRangeXPlug = PlugDescriptor("minRotateDampRangeX")
	minRotateDampRangeY_ : MinRotateDampRangeYPlug = PlugDescriptor("minRotateDampRangeY")
	ndy_ : MinRotateDampRangeYPlug = PlugDescriptor("minRotateDampRangeY")
	minRotateDampRangeZ_ : MinRotateDampRangeZPlug = PlugDescriptor("minRotateDampRangeZ")
	ndz_ : MinRotateDampRangeZPlug = PlugDescriptor("minRotateDampRangeZ")
	node : Joint = None
	pass
class MinRotateDampStrengthXPlug(Plug):
	parent : MinRotateDampStrengthPlug = PlugDescriptor("minRotateDampStrength")
	node : Joint = None
	pass
class MinRotateDampStrengthYPlug(Plug):
	parent : MinRotateDampStrengthPlug = PlugDescriptor("minRotateDampStrength")
	node : Joint = None
	pass
class MinRotateDampStrengthZPlug(Plug):
	parent : MinRotateDampStrengthPlug = PlugDescriptor("minRotateDampStrength")
	node : Joint = None
	pass
class MinRotateDampStrengthPlug(Plug):
	minRotateDampStrengthX_ : MinRotateDampStrengthXPlug = PlugDescriptor("minRotateDampStrengthX")
	nstx_ : MinRotateDampStrengthXPlug = PlugDescriptor("minRotateDampStrengthX")
	minRotateDampStrengthY_ : MinRotateDampStrengthYPlug = PlugDescriptor("minRotateDampStrengthY")
	nsty_ : MinRotateDampStrengthYPlug = PlugDescriptor("minRotateDampStrengthY")
	minRotateDampStrengthZ_ : MinRotateDampStrengthZPlug = PlugDescriptor("minRotateDampStrengthZ")
	nstz_ : MinRotateDampStrengthZPlug = PlugDescriptor("minRotateDampStrengthZ")
	node : Joint = None
	pass
class OtherTypePlug(Plug):
	node : Joint = None
	pass
class PreferredAngleXPlug(Plug):
	parent : PreferredAnglePlug = PlugDescriptor("preferredAngle")
	node : Joint = None
	pass
class PreferredAngleYPlug(Plug):
	parent : PreferredAnglePlug = PlugDescriptor("preferredAngle")
	node : Joint = None
	pass
class PreferredAngleZPlug(Plug):
	parent : PreferredAnglePlug = PlugDescriptor("preferredAngle")
	node : Joint = None
	pass
class PreferredAnglePlug(Plug):
	preferredAngleX_ : PreferredAngleXPlug = PlugDescriptor("preferredAngleX")
	pax_ : PreferredAngleXPlug = PlugDescriptor("preferredAngleX")
	preferredAngleY_ : PreferredAngleYPlug = PlugDescriptor("preferredAngleY")
	pay_ : PreferredAngleYPlug = PlugDescriptor("preferredAngleY")
	preferredAngleZ_ : PreferredAngleZPlug = PlugDescriptor("preferredAngleZ")
	paz_ : PreferredAngleZPlug = PlugDescriptor("preferredAngleZ")
	node : Joint = None
	pass
class RadiusPlug(Plug):
	node : Joint = None
	pass
class SegmentScaleCompensatePlug(Plug):
	node : Joint = None
	pass
class SidePlug(Plug):
	node : Joint = None
	pass
class StiffnessXPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : Joint = None
	pass
class StiffnessYPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : Joint = None
	pass
class StiffnessZPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : Joint = None
	pass
class StiffnessPlug(Plug):
	stiffnessX_ : StiffnessXPlug = PlugDescriptor("stiffnessX")
	stx_ : StiffnessXPlug = PlugDescriptor("stiffnessX")
	stiffnessY_ : StiffnessYPlug = PlugDescriptor("stiffnessY")
	sty_ : StiffnessYPlug = PlugDescriptor("stiffnessY")
	stiffnessZ_ : StiffnessZPlug = PlugDescriptor("stiffnessZ")
	stz_ : StiffnessZPlug = PlugDescriptor("stiffnessZ")
	node : Joint = None
	pass
class TypePlug(Plug):
	node : Joint = None
	pass
# endregion


# define node class
class Joint(Transform):
	bindInverseScaleX_ : BindInverseScaleXPlug = PlugDescriptor("bindInverseScaleX")
	bindInverseScaleY_ : BindInverseScaleYPlug = PlugDescriptor("bindInverseScaleY")
	bindInverseScaleZ_ : BindInverseScaleZPlug = PlugDescriptor("bindInverseScaleZ")
	bindInverseScale_ : BindInverseScalePlug = PlugDescriptor("bindInverseScale")
	bindJointOrientX_ : BindJointOrientXPlug = PlugDescriptor("bindJointOrientX")
	bindJointOrientY_ : BindJointOrientYPlug = PlugDescriptor("bindJointOrientY")
	bindJointOrientZ_ : BindJointOrientZPlug = PlugDescriptor("bindJointOrientZ")
	bindJointOrient_ : BindJointOrientPlug = PlugDescriptor("bindJointOrient")
	bindPose_ : BindPosePlug = PlugDescriptor("bindPose")
	bindRotateAxisX_ : BindRotateAxisXPlug = PlugDescriptor("bindRotateAxisX")
	bindRotateAxisY_ : BindRotateAxisYPlug = PlugDescriptor("bindRotateAxisY")
	bindRotateAxisZ_ : BindRotateAxisZPlug = PlugDescriptor("bindRotateAxisZ")
	bindRotateAxis_ : BindRotateAxisPlug = PlugDescriptor("bindRotateAxis")
	bindRotationX_ : BindRotationXPlug = PlugDescriptor("bindRotationX")
	bindRotationY_ : BindRotationYPlug = PlugDescriptor("bindRotationY")
	bindRotationZ_ : BindRotationZPlug = PlugDescriptor("bindRotationZ")
	bindRotation_ : BindRotationPlug = PlugDescriptor("bindRotation")
	bindScaleX_ : BindScaleXPlug = PlugDescriptor("bindScaleX")
	bindScaleY_ : BindScaleYPlug = PlugDescriptor("bindScaleY")
	bindScaleZ_ : BindScaleZPlug = PlugDescriptor("bindScaleZ")
	bindScale_ : BindScalePlug = PlugDescriptor("bindScale")
	bindSegmentScaleCompensate_ : BindSegmentScaleCompensatePlug = PlugDescriptor("bindSegmentScaleCompensate")
	dofMask_ : DofMaskPlug = PlugDescriptor("dofMask")
	drawLabel_ : DrawLabelPlug = PlugDescriptor("drawLabel")
	drawStyle_ : DrawStylePlug = PlugDescriptor("drawStyle")
	fkRotateX_ : FkRotateXPlug = PlugDescriptor("fkRotateX")
	fkRotateY_ : FkRotateYPlug = PlugDescriptor("fkRotateY")
	fkRotateZ_ : FkRotateZPlug = PlugDescriptor("fkRotateZ")
	fkRotate_ : FkRotatePlug = PlugDescriptor("fkRotate")
	hikFkJoint_ : HikFkJointPlug = PlugDescriptor("hikFkJoint")
	hikNodeID_ : HikNodeIDPlug = PlugDescriptor("hikNodeID")
	ikRotateX_ : IkRotateXPlug = PlugDescriptor("ikRotateX")
	ikRotateY_ : IkRotateYPlug = PlugDescriptor("ikRotateY")
	ikRotateZ_ : IkRotateZPlug = PlugDescriptor("ikRotateZ")
	ikRotate_ : IkRotatePlug = PlugDescriptor("ikRotate")
	inIKSolveFlag_ : InIKSolveFlagPlug = PlugDescriptor("inIKSolveFlag")
	inverseScaleX_ : InverseScaleXPlug = PlugDescriptor("inverseScaleX")
	inverseScaleY_ : InverseScaleYPlug = PlugDescriptor("inverseScaleY")
	inverseScaleZ_ : InverseScaleZPlug = PlugDescriptor("inverseScaleZ")
	inverseScale_ : InverseScalePlug = PlugDescriptor("inverseScale")
	isIKDirtyFlag_ : IsIKDirtyFlagPlug = PlugDescriptor("isIKDirtyFlag")
	jointOrientX_ : JointOrientXPlug = PlugDescriptor("jointOrientX")
	jointOrientY_ : JointOrientYPlug = PlugDescriptor("jointOrientY")
	jointOrientZ_ : JointOrientZPlug = PlugDescriptor("jointOrientZ")
	jointOrient_ : JointOrientPlug = PlugDescriptor("jointOrient")
	jointOrientType_ : JointOrientTypePlug = PlugDescriptor("jointOrientType")
	jointType_ : JointTypePlug = PlugDescriptor("jointType")
	jointTypeX_ : JointTypeXPlug = PlugDescriptor("jointTypeX")
	jointTypeY_ : JointTypeYPlug = PlugDescriptor("jointTypeY")
	jointTypeZ_ : JointTypeZPlug = PlugDescriptor("jointTypeZ")
	maxRotateDampRangeX_ : MaxRotateDampRangeXPlug = PlugDescriptor("maxRotateDampRangeX")
	maxRotateDampRangeY_ : MaxRotateDampRangeYPlug = PlugDescriptor("maxRotateDampRangeY")
	maxRotateDampRangeZ_ : MaxRotateDampRangeZPlug = PlugDescriptor("maxRotateDampRangeZ")
	maxRotateDampRange_ : MaxRotateDampRangePlug = PlugDescriptor("maxRotateDampRange")
	maxRotateDampStrengthX_ : MaxRotateDampStrengthXPlug = PlugDescriptor("maxRotateDampStrengthX")
	maxRotateDampStrengthY_ : MaxRotateDampStrengthYPlug = PlugDescriptor("maxRotateDampStrengthY")
	maxRotateDampStrengthZ_ : MaxRotateDampStrengthZPlug = PlugDescriptor("maxRotateDampStrengthZ")
	maxRotateDampStrength_ : MaxRotateDampStrengthPlug = PlugDescriptor("maxRotateDampStrength")
	minRotateDampRangeX_ : MinRotateDampRangeXPlug = PlugDescriptor("minRotateDampRangeX")
	minRotateDampRangeY_ : MinRotateDampRangeYPlug = PlugDescriptor("minRotateDampRangeY")
	minRotateDampRangeZ_ : MinRotateDampRangeZPlug = PlugDescriptor("minRotateDampRangeZ")
	minRotateDampRange_ : MinRotateDampRangePlug = PlugDescriptor("minRotateDampRange")
	minRotateDampStrengthX_ : MinRotateDampStrengthXPlug = PlugDescriptor("minRotateDampStrengthX")
	minRotateDampStrengthY_ : MinRotateDampStrengthYPlug = PlugDescriptor("minRotateDampStrengthY")
	minRotateDampStrengthZ_ : MinRotateDampStrengthZPlug = PlugDescriptor("minRotateDampStrengthZ")
	minRotateDampStrength_ : MinRotateDampStrengthPlug = PlugDescriptor("minRotateDampStrength")
	otherType_ : OtherTypePlug = PlugDescriptor("otherType")
	preferredAngleX_ : PreferredAngleXPlug = PlugDescriptor("preferredAngleX")
	preferredAngleY_ : PreferredAngleYPlug = PlugDescriptor("preferredAngleY")
	preferredAngleZ_ : PreferredAngleZPlug = PlugDescriptor("preferredAngleZ")
	preferredAngle_ : PreferredAnglePlug = PlugDescriptor("preferredAngle")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	segmentScaleCompensate_ : SegmentScaleCompensatePlug = PlugDescriptor("segmentScaleCompensate")
	side_ : SidePlug = PlugDescriptor("side")
	stiffnessX_ : StiffnessXPlug = PlugDescriptor("stiffnessX")
	stiffnessY_ : StiffnessYPlug = PlugDescriptor("stiffnessY")
	stiffnessZ_ : StiffnessZPlug = PlugDescriptor("stiffnessZ")
	stiffness_ : StiffnessPlug = PlugDescriptor("stiffness")
	type_ : TypePlug = PlugDescriptor("type")

	# node attributes

	typeName = "joint"
	apiTypeInt = 121
	apiTypeStr = "kJoint"
	typeIdInt = 1246710094
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["bindInverseScaleX", "bindInverseScaleY", "bindInverseScaleZ", "bindInverseScale", "bindJointOrientX", "bindJointOrientY", "bindJointOrientZ", "bindJointOrient", "bindPose", "bindRotateAxisX", "bindRotateAxisY", "bindRotateAxisZ", "bindRotateAxis", "bindRotationX", "bindRotationY", "bindRotationZ", "bindRotation", "bindScaleX", "bindScaleY", "bindScaleZ", "bindScale", "bindSegmentScaleCompensate", "dofMask", "drawLabel", "drawStyle", "fkRotateX", "fkRotateY", "fkRotateZ", "fkRotate", "hikFkJoint", "hikNodeID", "ikRotateX", "ikRotateY", "ikRotateZ", "ikRotate", "inIKSolveFlag", "inverseScaleX", "inverseScaleY", "inverseScaleZ", "inverseScale", "isIKDirtyFlag", "jointOrientX", "jointOrientY", "jointOrientZ", "jointOrient", "jointOrientType", "jointType", "jointTypeX", "jointTypeY", "jointTypeZ", "maxRotateDampRangeX", "maxRotateDampRangeY", "maxRotateDampRangeZ", "maxRotateDampRange", "maxRotateDampStrengthX", "maxRotateDampStrengthY", "maxRotateDampStrengthZ", "maxRotateDampStrength", "minRotateDampRangeX", "minRotateDampRangeY", "minRotateDampRangeZ", "minRotateDampRange", "minRotateDampStrengthX", "minRotateDampStrengthY", "minRotateDampStrengthZ", "minRotateDampStrength", "otherType", "preferredAngleX", "preferredAngleY", "preferredAngleZ", "preferredAngle", "radius", "segmentScaleCompensate", "side", "stiffnessX", "stiffnessY", "stiffnessZ", "stiffness", "type"]
	nodeLeafPlugs = ["bindInverseScale", "bindJointOrient", "bindPose", "bindRotateAxis", "bindRotation", "bindScale", "bindSegmentScaleCompensate", "dofMask", "drawLabel", "drawStyle", "fkRotate", "hikFkJoint", "hikNodeID", "ikRotate", "inIKSolveFlag", "inverseScale", "isIKDirtyFlag", "jointOrient", "jointOrientType", "jointType", "jointTypeX", "jointTypeY", "jointTypeZ", "maxRotateDampRange", "maxRotateDampStrength", "minRotateDampRange", "minRotateDampStrength", "otherType", "preferredAngle", "radius", "segmentScaleCompensate", "side", "stiffness", "type"]
	pass

