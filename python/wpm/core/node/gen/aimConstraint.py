

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Constraint = retriever.getNodeCls("Constraint")
assert Constraint
if T.TYPE_CHECKING:
	from .. import Constraint

# add node doc



# region plug type defs
class AimVectorXPlug(Plug):
	parent : AimVectorPlug = PlugDescriptor("aimVector")
	node : AimConstraint = None
	pass
class AimVectorYPlug(Plug):
	parent : AimVectorPlug = PlugDescriptor("aimVector")
	node : AimConstraint = None
	pass
class AimVectorZPlug(Plug):
	parent : AimVectorPlug = PlugDescriptor("aimVector")
	node : AimConstraint = None
	pass
class AimVectorPlug(Plug):
	aimVectorX_ : AimVectorXPlug = PlugDescriptor("aimVectorX")
	ax_ : AimVectorXPlug = PlugDescriptor("aimVectorX")
	aimVectorY_ : AimVectorYPlug = PlugDescriptor("aimVectorY")
	ay_ : AimVectorYPlug = PlugDescriptor("aimVectorY")
	aimVectorZ_ : AimVectorZPlug = PlugDescriptor("aimVectorZ")
	az_ : AimVectorZPlug = PlugDescriptor("aimVectorZ")
	node : AimConstraint = None
	pass
class ConstraintJointOrientXPlug(Plug):
	parent : ConstraintJointOrientPlug = PlugDescriptor("constraintJointOrient")
	node : AimConstraint = None
	pass
class ConstraintJointOrientYPlug(Plug):
	parent : ConstraintJointOrientPlug = PlugDescriptor("constraintJointOrient")
	node : AimConstraint = None
	pass
class ConstraintJointOrientZPlug(Plug):
	parent : ConstraintJointOrientPlug = PlugDescriptor("constraintJointOrient")
	node : AimConstraint = None
	pass
class ConstraintJointOrientPlug(Plug):
	constraintJointOrientX_ : ConstraintJointOrientXPlug = PlugDescriptor("constraintJointOrientX")
	cjox_ : ConstraintJointOrientXPlug = PlugDescriptor("constraintJointOrientX")
	constraintJointOrientY_ : ConstraintJointOrientYPlug = PlugDescriptor("constraintJointOrientY")
	cjoy_ : ConstraintJointOrientYPlug = PlugDescriptor("constraintJointOrientY")
	constraintJointOrientZ_ : ConstraintJointOrientZPlug = PlugDescriptor("constraintJointOrientZ")
	cjoz_ : ConstraintJointOrientZPlug = PlugDescriptor("constraintJointOrientZ")
	node : AimConstraint = None
	pass
class ConstraintParentInverseMatrixPlug(Plug):
	node : AimConstraint = None
	pass
class ConstraintRotateXPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : AimConstraint = None
	pass
class ConstraintRotateYPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : AimConstraint = None
	pass
class ConstraintRotateZPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : AimConstraint = None
	pass
class ConstraintRotatePlug(Plug):
	constraintRotateX_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	crx_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	constraintRotateY_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	cry_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	constraintRotateZ_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	crz_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	node : AimConstraint = None
	pass
class ConstraintRotateOrderPlug(Plug):
	node : AimConstraint = None
	pass
class ConstraintRotatePivotXPlug(Plug):
	parent : ConstraintRotatePivotPlug = PlugDescriptor("constraintRotatePivot")
	node : AimConstraint = None
	pass
class ConstraintRotatePivotYPlug(Plug):
	parent : ConstraintRotatePivotPlug = PlugDescriptor("constraintRotatePivot")
	node : AimConstraint = None
	pass
class ConstraintRotatePivotZPlug(Plug):
	parent : ConstraintRotatePivotPlug = PlugDescriptor("constraintRotatePivot")
	node : AimConstraint = None
	pass
class ConstraintRotatePivotPlug(Plug):
	constraintRotatePivotX_ : ConstraintRotatePivotXPlug = PlugDescriptor("constraintRotatePivotX")
	crpx_ : ConstraintRotatePivotXPlug = PlugDescriptor("constraintRotatePivotX")
	constraintRotatePivotY_ : ConstraintRotatePivotYPlug = PlugDescriptor("constraintRotatePivotY")
	crpy_ : ConstraintRotatePivotYPlug = PlugDescriptor("constraintRotatePivotY")
	constraintRotatePivotZ_ : ConstraintRotatePivotZPlug = PlugDescriptor("constraintRotatePivotZ")
	crpz_ : ConstraintRotatePivotZPlug = PlugDescriptor("constraintRotatePivotZ")
	node : AimConstraint = None
	pass
class ConstraintRotateTranslateXPlug(Plug):
	parent : ConstraintRotateTranslatePlug = PlugDescriptor("constraintRotateTranslate")
	node : AimConstraint = None
	pass
class ConstraintRotateTranslateYPlug(Plug):
	parent : ConstraintRotateTranslatePlug = PlugDescriptor("constraintRotateTranslate")
	node : AimConstraint = None
	pass
class ConstraintRotateTranslateZPlug(Plug):
	parent : ConstraintRotateTranslatePlug = PlugDescriptor("constraintRotateTranslate")
	node : AimConstraint = None
	pass
class ConstraintRotateTranslatePlug(Plug):
	constraintRotateTranslateX_ : ConstraintRotateTranslateXPlug = PlugDescriptor("constraintRotateTranslateX")
	crtx_ : ConstraintRotateTranslateXPlug = PlugDescriptor("constraintRotateTranslateX")
	constraintRotateTranslateY_ : ConstraintRotateTranslateYPlug = PlugDescriptor("constraintRotateTranslateY")
	crty_ : ConstraintRotateTranslateYPlug = PlugDescriptor("constraintRotateTranslateY")
	constraintRotateTranslateZ_ : ConstraintRotateTranslateZPlug = PlugDescriptor("constraintRotateTranslateZ")
	crtz_ : ConstraintRotateTranslateZPlug = PlugDescriptor("constraintRotateTranslateZ")
	node : AimConstraint = None
	pass
class ConstraintTranslateXPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : AimConstraint = None
	pass
class ConstraintTranslateYPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : AimConstraint = None
	pass
class ConstraintTranslateZPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : AimConstraint = None
	pass
class ConstraintTranslatePlug(Plug):
	constraintTranslateX_ : ConstraintTranslateXPlug = PlugDescriptor("constraintTranslateX")
	ctx_ : ConstraintTranslateXPlug = PlugDescriptor("constraintTranslateX")
	constraintTranslateY_ : ConstraintTranslateYPlug = PlugDescriptor("constraintTranslateY")
	cty_ : ConstraintTranslateYPlug = PlugDescriptor("constraintTranslateY")
	constraintTranslateZ_ : ConstraintTranslateZPlug = PlugDescriptor("constraintTranslateZ")
	ctz_ : ConstraintTranslateZPlug = PlugDescriptor("constraintTranslateZ")
	node : AimConstraint = None
	pass
class ConstraintVectorXPlug(Plug):
	parent : ConstraintVectorPlug = PlugDescriptor("constraintVector")
	node : AimConstraint = None
	pass
class ConstraintVectorYPlug(Plug):
	parent : ConstraintVectorPlug = PlugDescriptor("constraintVector")
	node : AimConstraint = None
	pass
class ConstraintVectorZPlug(Plug):
	parent : ConstraintVectorPlug = PlugDescriptor("constraintVector")
	node : AimConstraint = None
	pass
class ConstraintVectorPlug(Plug):
	constraintVectorX_ : ConstraintVectorXPlug = PlugDescriptor("constraintVectorX")
	cvx_ : ConstraintVectorXPlug = PlugDescriptor("constraintVectorX")
	constraintVectorY_ : ConstraintVectorYPlug = PlugDescriptor("constraintVectorY")
	cvy_ : ConstraintVectorYPlug = PlugDescriptor("constraintVectorY")
	constraintVectorZ_ : ConstraintVectorZPlug = PlugDescriptor("constraintVectorZ")
	cvz_ : ConstraintVectorZPlug = PlugDescriptor("constraintVectorZ")
	node : AimConstraint = None
	pass
class InverseScaleXPlug(Plug):
	parent : InverseScalePlug = PlugDescriptor("inverseScale")
	node : AimConstraint = None
	pass
class InverseScaleYPlug(Plug):
	parent : InverseScalePlug = PlugDescriptor("inverseScale")
	node : AimConstraint = None
	pass
class InverseScaleZPlug(Plug):
	parent : InverseScalePlug = PlugDescriptor("inverseScale")
	node : AimConstraint = None
	pass
class InverseScalePlug(Plug):
	inverseScaleX_ : InverseScaleXPlug = PlugDescriptor("inverseScaleX")
	isx_ : InverseScaleXPlug = PlugDescriptor("inverseScaleX")
	inverseScaleY_ : InverseScaleYPlug = PlugDescriptor("inverseScaleY")
	isy_ : InverseScaleYPlug = PlugDescriptor("inverseScaleY")
	inverseScaleZ_ : InverseScaleZPlug = PlugDescriptor("inverseScaleZ")
	isz_ : InverseScaleZPlug = PlugDescriptor("inverseScaleZ")
	node : AimConstraint = None
	pass
class OffsetXPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : AimConstraint = None
	pass
class OffsetYPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : AimConstraint = None
	pass
class OffsetZPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : AimConstraint = None
	pass
class OffsetPlug(Plug):
	offsetX_ : OffsetXPlug = PlugDescriptor("offsetX")
	ox_ : OffsetXPlug = PlugDescriptor("offsetX")
	offsetY_ : OffsetYPlug = PlugDescriptor("offsetY")
	oy_ : OffsetYPlug = PlugDescriptor("offsetY")
	offsetZ_ : OffsetZPlug = PlugDescriptor("offsetZ")
	oz_ : OffsetZPlug = PlugDescriptor("offsetZ")
	node : AimConstraint = None
	pass
class RestRotateXPlug(Plug):
	parent : RestRotatePlug = PlugDescriptor("restRotate")
	node : AimConstraint = None
	pass
class RestRotateYPlug(Plug):
	parent : RestRotatePlug = PlugDescriptor("restRotate")
	node : AimConstraint = None
	pass
class RestRotateZPlug(Plug):
	parent : RestRotatePlug = PlugDescriptor("restRotate")
	node : AimConstraint = None
	pass
class RestRotatePlug(Plug):
	restRotateX_ : RestRotateXPlug = PlugDescriptor("restRotateX")
	rrx_ : RestRotateXPlug = PlugDescriptor("restRotateX")
	restRotateY_ : RestRotateYPlug = PlugDescriptor("restRotateY")
	rry_ : RestRotateYPlug = PlugDescriptor("restRotateY")
	restRotateZ_ : RestRotateZPlug = PlugDescriptor("restRotateZ")
	rrz_ : RestRotateZPlug = PlugDescriptor("restRotateZ")
	node : AimConstraint = None
	pass
class ScaleCompensatePlug(Plug):
	node : AimConstraint = None
	pass
class TargetParentMatrixPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : AimConstraint = None
	pass
class TargetRotatePivotXPlug(Plug):
	parent : TargetRotatePivotPlug = PlugDescriptor("targetRotatePivot")
	node : AimConstraint = None
	pass
class TargetRotatePivotYPlug(Plug):
	parent : TargetRotatePivotPlug = PlugDescriptor("targetRotatePivot")
	node : AimConstraint = None
	pass
class TargetRotatePivotZPlug(Plug):
	parent : TargetRotatePivotPlug = PlugDescriptor("targetRotatePivot")
	node : AimConstraint = None
	pass
class TargetRotatePivotPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetRotatePivotX_ : TargetRotatePivotXPlug = PlugDescriptor("targetRotatePivotX")
	trpx_ : TargetRotatePivotXPlug = PlugDescriptor("targetRotatePivotX")
	targetRotatePivotY_ : TargetRotatePivotYPlug = PlugDescriptor("targetRotatePivotY")
	trpy_ : TargetRotatePivotYPlug = PlugDescriptor("targetRotatePivotY")
	targetRotatePivotZ_ : TargetRotatePivotZPlug = PlugDescriptor("targetRotatePivotZ")
	trpz_ : TargetRotatePivotZPlug = PlugDescriptor("targetRotatePivotZ")
	node : AimConstraint = None
	pass
class TargetRotateTranslateXPlug(Plug):
	parent : TargetRotateTranslatePlug = PlugDescriptor("targetRotateTranslate")
	node : AimConstraint = None
	pass
class TargetRotateTranslateYPlug(Plug):
	parent : TargetRotateTranslatePlug = PlugDescriptor("targetRotateTranslate")
	node : AimConstraint = None
	pass
class TargetRotateTranslateZPlug(Plug):
	parent : TargetRotateTranslatePlug = PlugDescriptor("targetRotateTranslate")
	node : AimConstraint = None
	pass
class TargetRotateTranslatePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetRotateTranslateX_ : TargetRotateTranslateXPlug = PlugDescriptor("targetRotateTranslateX")
	trtx_ : TargetRotateTranslateXPlug = PlugDescriptor("targetRotateTranslateX")
	targetRotateTranslateY_ : TargetRotateTranslateYPlug = PlugDescriptor("targetRotateTranslateY")
	trty_ : TargetRotateTranslateYPlug = PlugDescriptor("targetRotateTranslateY")
	targetRotateTranslateZ_ : TargetRotateTranslateZPlug = PlugDescriptor("targetRotateTranslateZ")
	trtz_ : TargetRotateTranslateZPlug = PlugDescriptor("targetRotateTranslateZ")
	node : AimConstraint = None
	pass
class TargetTranslateXPlug(Plug):
	parent : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	node : AimConstraint = None
	pass
class TargetTranslateYPlug(Plug):
	parent : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	node : AimConstraint = None
	pass
class TargetTranslateZPlug(Plug):
	parent : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	node : AimConstraint = None
	pass
class TargetTranslatePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetTranslateX_ : TargetTranslateXPlug = PlugDescriptor("targetTranslateX")
	ttx_ : TargetTranslateXPlug = PlugDescriptor("targetTranslateX")
	targetTranslateY_ : TargetTranslateYPlug = PlugDescriptor("targetTranslateY")
	tty_ : TargetTranslateYPlug = PlugDescriptor("targetTranslateY")
	targetTranslateZ_ : TargetTranslateZPlug = PlugDescriptor("targetTranslateZ")
	ttz_ : TargetTranslateZPlug = PlugDescriptor("targetTranslateZ")
	node : AimConstraint = None
	pass
class TargetWeightPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : AimConstraint = None
	pass
class TargetPlug(Plug):
	targetParentMatrix_ : TargetParentMatrixPlug = PlugDescriptor("targetParentMatrix")
	tpm_ : TargetParentMatrixPlug = PlugDescriptor("targetParentMatrix")
	targetRotatePivot_ : TargetRotatePivotPlug = PlugDescriptor("targetRotatePivot")
	trp_ : TargetRotatePivotPlug = PlugDescriptor("targetRotatePivot")
	targetRotateTranslate_ : TargetRotateTranslatePlug = PlugDescriptor("targetRotateTranslate")
	trt_ : TargetRotateTranslatePlug = PlugDescriptor("targetRotateTranslate")
	targetTranslate_ : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	tt_ : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	targetWeight_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	tw_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	node : AimConstraint = None
	pass
class UpVectorXPlug(Plug):
	parent : UpVectorPlug = PlugDescriptor("upVector")
	node : AimConstraint = None
	pass
class UpVectorYPlug(Plug):
	parent : UpVectorPlug = PlugDescriptor("upVector")
	node : AimConstraint = None
	pass
class UpVectorZPlug(Plug):
	parent : UpVectorPlug = PlugDescriptor("upVector")
	node : AimConstraint = None
	pass
class UpVectorPlug(Plug):
	upVectorX_ : UpVectorXPlug = PlugDescriptor("upVectorX")
	ux_ : UpVectorXPlug = PlugDescriptor("upVectorX")
	upVectorY_ : UpVectorYPlug = PlugDescriptor("upVectorY")
	uy_ : UpVectorYPlug = PlugDescriptor("upVectorY")
	upVectorZ_ : UpVectorZPlug = PlugDescriptor("upVectorZ")
	uz_ : UpVectorZPlug = PlugDescriptor("upVectorZ")
	node : AimConstraint = None
	pass
class UseOldOffsetCalculationPlug(Plug):
	node : AimConstraint = None
	pass
class WorldUpMatrixPlug(Plug):
	node : AimConstraint = None
	pass
class WorldUpTypePlug(Plug):
	node : AimConstraint = None
	pass
class WorldUpVectorXPlug(Plug):
	parent : WorldUpVectorPlug = PlugDescriptor("worldUpVector")
	node : AimConstraint = None
	pass
class WorldUpVectorYPlug(Plug):
	parent : WorldUpVectorPlug = PlugDescriptor("worldUpVector")
	node : AimConstraint = None
	pass
class WorldUpVectorZPlug(Plug):
	parent : WorldUpVectorPlug = PlugDescriptor("worldUpVector")
	node : AimConstraint = None
	pass
class WorldUpVectorPlug(Plug):
	worldUpVectorX_ : WorldUpVectorXPlug = PlugDescriptor("worldUpVectorX")
	wux_ : WorldUpVectorXPlug = PlugDescriptor("worldUpVectorX")
	worldUpVectorY_ : WorldUpVectorYPlug = PlugDescriptor("worldUpVectorY")
	wuy_ : WorldUpVectorYPlug = PlugDescriptor("worldUpVectorY")
	worldUpVectorZ_ : WorldUpVectorZPlug = PlugDescriptor("worldUpVectorZ")
	wuz_ : WorldUpVectorZPlug = PlugDescriptor("worldUpVectorZ")
	node : AimConstraint = None
	pass
# endregion


# define node class
class AimConstraint(Constraint):
	aimVectorX_ : AimVectorXPlug = PlugDescriptor("aimVectorX")
	aimVectorY_ : AimVectorYPlug = PlugDescriptor("aimVectorY")
	aimVectorZ_ : AimVectorZPlug = PlugDescriptor("aimVectorZ")
	aimVector_ : AimVectorPlug = PlugDescriptor("aimVector")
	constraintJointOrientX_ : ConstraintJointOrientXPlug = PlugDescriptor("constraintJointOrientX")
	constraintJointOrientY_ : ConstraintJointOrientYPlug = PlugDescriptor("constraintJointOrientY")
	constraintJointOrientZ_ : ConstraintJointOrientZPlug = PlugDescriptor("constraintJointOrientZ")
	constraintJointOrient_ : ConstraintJointOrientPlug = PlugDescriptor("constraintJointOrient")
	constraintParentInverseMatrix_ : ConstraintParentInverseMatrixPlug = PlugDescriptor("constraintParentInverseMatrix")
	constraintRotateX_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	constraintRotateY_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	constraintRotateZ_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	constraintRotate_ : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	constraintRotateOrder_ : ConstraintRotateOrderPlug = PlugDescriptor("constraintRotateOrder")
	constraintRotatePivotX_ : ConstraintRotatePivotXPlug = PlugDescriptor("constraintRotatePivotX")
	constraintRotatePivotY_ : ConstraintRotatePivotYPlug = PlugDescriptor("constraintRotatePivotY")
	constraintRotatePivotZ_ : ConstraintRotatePivotZPlug = PlugDescriptor("constraintRotatePivotZ")
	constraintRotatePivot_ : ConstraintRotatePivotPlug = PlugDescriptor("constraintRotatePivot")
	constraintRotateTranslateX_ : ConstraintRotateTranslateXPlug = PlugDescriptor("constraintRotateTranslateX")
	constraintRotateTranslateY_ : ConstraintRotateTranslateYPlug = PlugDescriptor("constraintRotateTranslateY")
	constraintRotateTranslateZ_ : ConstraintRotateTranslateZPlug = PlugDescriptor("constraintRotateTranslateZ")
	constraintRotateTranslate_ : ConstraintRotateTranslatePlug = PlugDescriptor("constraintRotateTranslate")
	constraintTranslateX_ : ConstraintTranslateXPlug = PlugDescriptor("constraintTranslateX")
	constraintTranslateY_ : ConstraintTranslateYPlug = PlugDescriptor("constraintTranslateY")
	constraintTranslateZ_ : ConstraintTranslateZPlug = PlugDescriptor("constraintTranslateZ")
	constraintTranslate_ : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	constraintVectorX_ : ConstraintVectorXPlug = PlugDescriptor("constraintVectorX")
	constraintVectorY_ : ConstraintVectorYPlug = PlugDescriptor("constraintVectorY")
	constraintVectorZ_ : ConstraintVectorZPlug = PlugDescriptor("constraintVectorZ")
	constraintVector_ : ConstraintVectorPlug = PlugDescriptor("constraintVector")
	inverseScaleX_ : InverseScaleXPlug = PlugDescriptor("inverseScaleX")
	inverseScaleY_ : InverseScaleYPlug = PlugDescriptor("inverseScaleY")
	inverseScaleZ_ : InverseScaleZPlug = PlugDescriptor("inverseScaleZ")
	inverseScale_ : InverseScalePlug = PlugDescriptor("inverseScale")
	offsetX_ : OffsetXPlug = PlugDescriptor("offsetX")
	offsetY_ : OffsetYPlug = PlugDescriptor("offsetY")
	offsetZ_ : OffsetZPlug = PlugDescriptor("offsetZ")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	restRotateX_ : RestRotateXPlug = PlugDescriptor("restRotateX")
	restRotateY_ : RestRotateYPlug = PlugDescriptor("restRotateY")
	restRotateZ_ : RestRotateZPlug = PlugDescriptor("restRotateZ")
	restRotate_ : RestRotatePlug = PlugDescriptor("restRotate")
	scaleCompensate_ : ScaleCompensatePlug = PlugDescriptor("scaleCompensate")
	targetParentMatrix_ : TargetParentMatrixPlug = PlugDescriptor("targetParentMatrix")
	targetRotatePivotX_ : TargetRotatePivotXPlug = PlugDescriptor("targetRotatePivotX")
	targetRotatePivotY_ : TargetRotatePivotYPlug = PlugDescriptor("targetRotatePivotY")
	targetRotatePivotZ_ : TargetRotatePivotZPlug = PlugDescriptor("targetRotatePivotZ")
	targetRotatePivot_ : TargetRotatePivotPlug = PlugDescriptor("targetRotatePivot")
	targetRotateTranslateX_ : TargetRotateTranslateXPlug = PlugDescriptor("targetRotateTranslateX")
	targetRotateTranslateY_ : TargetRotateTranslateYPlug = PlugDescriptor("targetRotateTranslateY")
	targetRotateTranslateZ_ : TargetRotateTranslateZPlug = PlugDescriptor("targetRotateTranslateZ")
	targetRotateTranslate_ : TargetRotateTranslatePlug = PlugDescriptor("targetRotateTranslate")
	targetTranslateX_ : TargetTranslateXPlug = PlugDescriptor("targetTranslateX")
	targetTranslateY_ : TargetTranslateYPlug = PlugDescriptor("targetTranslateY")
	targetTranslateZ_ : TargetTranslateZPlug = PlugDescriptor("targetTranslateZ")
	targetTranslate_ : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	targetWeight_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	target_ : TargetPlug = PlugDescriptor("target")
	upVectorX_ : UpVectorXPlug = PlugDescriptor("upVectorX")
	upVectorY_ : UpVectorYPlug = PlugDescriptor("upVectorY")
	upVectorZ_ : UpVectorZPlug = PlugDescriptor("upVectorZ")
	upVector_ : UpVectorPlug = PlugDescriptor("upVector")
	useOldOffsetCalculation_ : UseOldOffsetCalculationPlug = PlugDescriptor("useOldOffsetCalculation")
	worldUpMatrix_ : WorldUpMatrixPlug = PlugDescriptor("worldUpMatrix")
	worldUpType_ : WorldUpTypePlug = PlugDescriptor("worldUpType")
	worldUpVectorX_ : WorldUpVectorXPlug = PlugDescriptor("worldUpVectorX")
	worldUpVectorY_ : WorldUpVectorYPlug = PlugDescriptor("worldUpVectorY")
	worldUpVectorZ_ : WorldUpVectorZPlug = PlugDescriptor("worldUpVectorZ")
	worldUpVector_ : WorldUpVectorPlug = PlugDescriptor("worldUpVector")

	# node attributes

	typeName = "aimConstraint"
	apiTypeInt = 111
	apiTypeStr = "kAimConstraint"
	typeIdInt = 1145130307
	MFnCls = om.MFnTransform
	pass

