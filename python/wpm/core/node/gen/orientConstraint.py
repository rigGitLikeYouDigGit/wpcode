

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
class ConstraintJointOrientXPlug(Plug):
	parent : ConstraintJointOrientPlug = PlugDescriptor("constraintJointOrient")
	node : OrientConstraint = None
	pass
class ConstraintJointOrientYPlug(Plug):
	parent : ConstraintJointOrientPlug = PlugDescriptor("constraintJointOrient")
	node : OrientConstraint = None
	pass
class ConstraintJointOrientZPlug(Plug):
	parent : ConstraintJointOrientPlug = PlugDescriptor("constraintJointOrient")
	node : OrientConstraint = None
	pass
class ConstraintJointOrientPlug(Plug):
	constraintJointOrientX_ : ConstraintJointOrientXPlug = PlugDescriptor("constraintJointOrientX")
	cjox_ : ConstraintJointOrientXPlug = PlugDescriptor("constraintJointOrientX")
	constraintJointOrientY_ : ConstraintJointOrientYPlug = PlugDescriptor("constraintJointOrientY")
	cjoy_ : ConstraintJointOrientYPlug = PlugDescriptor("constraintJointOrientY")
	constraintJointOrientZ_ : ConstraintJointOrientZPlug = PlugDescriptor("constraintJointOrientZ")
	cjoz_ : ConstraintJointOrientZPlug = PlugDescriptor("constraintJointOrientZ")
	node : OrientConstraint = None
	pass
class ConstraintParentInverseMatrixPlug(Plug):
	node : OrientConstraint = None
	pass
class ConstraintRotateXPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : OrientConstraint = None
	pass
class ConstraintRotateYPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : OrientConstraint = None
	pass
class ConstraintRotateZPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : OrientConstraint = None
	pass
class ConstraintRotatePlug(Plug):
	constraintRotateX_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	crx_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	constraintRotateY_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	cry_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	constraintRotateZ_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	crz_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	node : OrientConstraint = None
	pass
class ConstraintRotateOrderPlug(Plug):
	node : OrientConstraint = None
	pass
class InterpCachePlug(Plug):
	node : OrientConstraint = None
	pass
class InterpTypePlug(Plug):
	node : OrientConstraint = None
	pass
class InverseScaleXPlug(Plug):
	parent : InverseScalePlug = PlugDescriptor("inverseScale")
	node : OrientConstraint = None
	pass
class InverseScaleYPlug(Plug):
	parent : InverseScalePlug = PlugDescriptor("inverseScale")
	node : OrientConstraint = None
	pass
class InverseScaleZPlug(Plug):
	parent : InverseScalePlug = PlugDescriptor("inverseScale")
	node : OrientConstraint = None
	pass
class InverseScalePlug(Plug):
	inverseScaleX_ : InverseScaleXPlug = PlugDescriptor("inverseScaleX")
	isx_ : InverseScaleXPlug = PlugDescriptor("inverseScaleX")
	inverseScaleY_ : InverseScaleYPlug = PlugDescriptor("inverseScaleY")
	isy_ : InverseScaleYPlug = PlugDescriptor("inverseScaleY")
	inverseScaleZ_ : InverseScaleZPlug = PlugDescriptor("inverseScaleZ")
	isz_ : InverseScaleZPlug = PlugDescriptor("inverseScaleZ")
	node : OrientConstraint = None
	pass
class LastTargetRotateXPlug(Plug):
	parent : LastTargetRotatePlug = PlugDescriptor("lastTargetRotate")
	node : OrientConstraint = None
	pass
class LastTargetRotateYPlug(Plug):
	parent : LastTargetRotatePlug = PlugDescriptor("lastTargetRotate")
	node : OrientConstraint = None
	pass
class LastTargetRotateZPlug(Plug):
	parent : LastTargetRotatePlug = PlugDescriptor("lastTargetRotate")
	node : OrientConstraint = None
	pass
class LastTargetRotatePlug(Plug):
	lastTargetRotateX_ : LastTargetRotateXPlug = PlugDescriptor("lastTargetRotateX")
	lrx_ : LastTargetRotateXPlug = PlugDescriptor("lastTargetRotateX")
	lastTargetRotateY_ : LastTargetRotateYPlug = PlugDescriptor("lastTargetRotateY")
	lry_ : LastTargetRotateYPlug = PlugDescriptor("lastTargetRotateY")
	lastTargetRotateZ_ : LastTargetRotateZPlug = PlugDescriptor("lastTargetRotateZ")
	lrz_ : LastTargetRotateZPlug = PlugDescriptor("lastTargetRotateZ")
	node : OrientConstraint = None
	pass
class OffsetXPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : OrientConstraint = None
	pass
class OffsetYPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : OrientConstraint = None
	pass
class OffsetZPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : OrientConstraint = None
	pass
class OffsetPlug(Plug):
	offsetX_ : OffsetXPlug = PlugDescriptor("offsetX")
	ox_ : OffsetXPlug = PlugDescriptor("offsetX")
	offsetY_ : OffsetYPlug = PlugDescriptor("offsetY")
	oy_ : OffsetYPlug = PlugDescriptor("offsetY")
	offsetZ_ : OffsetZPlug = PlugDescriptor("offsetZ")
	oz_ : OffsetZPlug = PlugDescriptor("offsetZ")
	node : OrientConstraint = None
	pass
class RestRotateXPlug(Plug):
	parent : RestRotatePlug = PlugDescriptor("restRotate")
	node : OrientConstraint = None
	pass
class RestRotateYPlug(Plug):
	parent : RestRotatePlug = PlugDescriptor("restRotate")
	node : OrientConstraint = None
	pass
class RestRotateZPlug(Plug):
	parent : RestRotatePlug = PlugDescriptor("restRotate")
	node : OrientConstraint = None
	pass
class RestRotatePlug(Plug):
	restRotateX_ : RestRotateXPlug = PlugDescriptor("restRotateX")
	rrx_ : RestRotateXPlug = PlugDescriptor("restRotateX")
	restRotateY_ : RestRotateYPlug = PlugDescriptor("restRotateY")
	rry_ : RestRotateYPlug = PlugDescriptor("restRotateY")
	restRotateZ_ : RestRotateZPlug = PlugDescriptor("restRotateZ")
	rrz_ : RestRotateZPlug = PlugDescriptor("restRotateZ")
	node : OrientConstraint = None
	pass
class ScaleCompensatePlug(Plug):
	node : OrientConstraint = None
	pass
class TargetJointOrientXPlug(Plug):
	parent : TargetJointOrientPlug = PlugDescriptor("targetJointOrient")
	node : OrientConstraint = None
	pass
class TargetJointOrientYPlug(Plug):
	parent : TargetJointOrientPlug = PlugDescriptor("targetJointOrient")
	node : OrientConstraint = None
	pass
class TargetJointOrientZPlug(Plug):
	parent : TargetJointOrientPlug = PlugDescriptor("targetJointOrient")
	node : OrientConstraint = None
	pass
class TargetJointOrientPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetJointOrientX_ : TargetJointOrientXPlug = PlugDescriptor("targetJointOrientX")
	tjox_ : TargetJointOrientXPlug = PlugDescriptor("targetJointOrientX")
	targetJointOrientY_ : TargetJointOrientYPlug = PlugDescriptor("targetJointOrientY")
	tjoy_ : TargetJointOrientYPlug = PlugDescriptor("targetJointOrientY")
	targetJointOrientZ_ : TargetJointOrientZPlug = PlugDescriptor("targetJointOrientZ")
	tjoz_ : TargetJointOrientZPlug = PlugDescriptor("targetJointOrientZ")
	node : OrientConstraint = None
	pass
class TargetParentMatrixPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : OrientConstraint = None
	pass
class TargetRotateXPlug(Plug):
	parent : TargetRotatePlug = PlugDescriptor("targetRotate")
	node : OrientConstraint = None
	pass
class TargetRotateYPlug(Plug):
	parent : TargetRotatePlug = PlugDescriptor("targetRotate")
	node : OrientConstraint = None
	pass
class TargetRotateZPlug(Plug):
	parent : TargetRotatePlug = PlugDescriptor("targetRotate")
	node : OrientConstraint = None
	pass
class TargetRotatePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetRotateX_ : TargetRotateXPlug = PlugDescriptor("targetRotateX")
	trx_ : TargetRotateXPlug = PlugDescriptor("targetRotateX")
	targetRotateY_ : TargetRotateYPlug = PlugDescriptor("targetRotateY")
	try_ : TargetRotateYPlug = PlugDescriptor("targetRotateY")
	targetRotateZ_ : TargetRotateZPlug = PlugDescriptor("targetRotateZ")
	trz_ : TargetRotateZPlug = PlugDescriptor("targetRotateZ")
	node : OrientConstraint = None
	pass
class TargetRotateCachedXPlug(Plug):
	parent : TargetRotateCachedPlug = PlugDescriptor("targetRotateCached")
	node : OrientConstraint = None
	pass
class TargetRotateCachedYPlug(Plug):
	parent : TargetRotateCachedPlug = PlugDescriptor("targetRotateCached")
	node : OrientConstraint = None
	pass
class TargetRotateCachedZPlug(Plug):
	parent : TargetRotateCachedPlug = PlugDescriptor("targetRotateCached")
	node : OrientConstraint = None
	pass
class TargetRotateCachedPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetRotateCachedX_ : TargetRotateCachedXPlug = PlugDescriptor("targetRotateCachedX")
	ctrx_ : TargetRotateCachedXPlug = PlugDescriptor("targetRotateCachedX")
	targetRotateCachedY_ : TargetRotateCachedYPlug = PlugDescriptor("targetRotateCachedY")
	ctry_ : TargetRotateCachedYPlug = PlugDescriptor("targetRotateCachedY")
	targetRotateCachedZ_ : TargetRotateCachedZPlug = PlugDescriptor("targetRotateCachedZ")
	ctrz_ : TargetRotateCachedZPlug = PlugDescriptor("targetRotateCachedZ")
	node : OrientConstraint = None
	pass
class TargetRotateOrderPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : OrientConstraint = None
	pass
class TargetWeightPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : OrientConstraint = None
	pass
class TargetPlug(Plug):
	targetJointOrient_ : TargetJointOrientPlug = PlugDescriptor("targetJointOrient")
	tjo_ : TargetJointOrientPlug = PlugDescriptor("targetJointOrient")
	targetParentMatrix_ : TargetParentMatrixPlug = PlugDescriptor("targetParentMatrix")
	tpm_ : TargetParentMatrixPlug = PlugDescriptor("targetParentMatrix")
	targetRotate_ : TargetRotatePlug = PlugDescriptor("targetRotate")
	tr_ : TargetRotatePlug = PlugDescriptor("targetRotate")
	targetRotateCached_ : TargetRotateCachedPlug = PlugDescriptor("targetRotateCached")
	ctr_ : TargetRotateCachedPlug = PlugDescriptor("targetRotateCached")
	targetRotateOrder_ : TargetRotateOrderPlug = PlugDescriptor("targetRotateOrder")
	tro_ : TargetRotateOrderPlug = PlugDescriptor("targetRotateOrder")
	targetWeight_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	tw_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	node : OrientConstraint = None
	pass
class UseOldOffsetCalculationPlug(Plug):
	node : OrientConstraint = None
	pass
# endregion


# define node class
class OrientConstraint(Constraint):
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
	interpCache_ : InterpCachePlug = PlugDescriptor("interpCache")
	interpType_ : InterpTypePlug = PlugDescriptor("interpType")
	inverseScaleX_ : InverseScaleXPlug = PlugDescriptor("inverseScaleX")
	inverseScaleY_ : InverseScaleYPlug = PlugDescriptor("inverseScaleY")
	inverseScaleZ_ : InverseScaleZPlug = PlugDescriptor("inverseScaleZ")
	inverseScale_ : InverseScalePlug = PlugDescriptor("inverseScale")
	lastTargetRotateX_ : LastTargetRotateXPlug = PlugDescriptor("lastTargetRotateX")
	lastTargetRotateY_ : LastTargetRotateYPlug = PlugDescriptor("lastTargetRotateY")
	lastTargetRotateZ_ : LastTargetRotateZPlug = PlugDescriptor("lastTargetRotateZ")
	lastTargetRotate_ : LastTargetRotatePlug = PlugDescriptor("lastTargetRotate")
	offsetX_ : OffsetXPlug = PlugDescriptor("offsetX")
	offsetY_ : OffsetYPlug = PlugDescriptor("offsetY")
	offsetZ_ : OffsetZPlug = PlugDescriptor("offsetZ")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	restRotateX_ : RestRotateXPlug = PlugDescriptor("restRotateX")
	restRotateY_ : RestRotateYPlug = PlugDescriptor("restRotateY")
	restRotateZ_ : RestRotateZPlug = PlugDescriptor("restRotateZ")
	restRotate_ : RestRotatePlug = PlugDescriptor("restRotate")
	scaleCompensate_ : ScaleCompensatePlug = PlugDescriptor("scaleCompensate")
	targetJointOrientX_ : TargetJointOrientXPlug = PlugDescriptor("targetJointOrientX")
	targetJointOrientY_ : TargetJointOrientYPlug = PlugDescriptor("targetJointOrientY")
	targetJointOrientZ_ : TargetJointOrientZPlug = PlugDescriptor("targetJointOrientZ")
	targetJointOrient_ : TargetJointOrientPlug = PlugDescriptor("targetJointOrient")
	targetParentMatrix_ : TargetParentMatrixPlug = PlugDescriptor("targetParentMatrix")
	targetRotateX_ : TargetRotateXPlug = PlugDescriptor("targetRotateX")
	targetRotateY_ : TargetRotateYPlug = PlugDescriptor("targetRotateY")
	targetRotateZ_ : TargetRotateZPlug = PlugDescriptor("targetRotateZ")
	targetRotate_ : TargetRotatePlug = PlugDescriptor("targetRotate")
	targetRotateCachedX_ : TargetRotateCachedXPlug = PlugDescriptor("targetRotateCachedX")
	targetRotateCachedY_ : TargetRotateCachedYPlug = PlugDescriptor("targetRotateCachedY")
	targetRotateCachedZ_ : TargetRotateCachedZPlug = PlugDescriptor("targetRotateCachedZ")
	targetRotateCached_ : TargetRotateCachedPlug = PlugDescriptor("targetRotateCached")
	targetRotateOrder_ : TargetRotateOrderPlug = PlugDescriptor("targetRotateOrder")
	targetWeight_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	target_ : TargetPlug = PlugDescriptor("target")
	useOldOffsetCalculation_ : UseOldOffsetCalculationPlug = PlugDescriptor("useOldOffsetCalculation")

	# node attributes

	typeName = "orientConstraint"
	apiTypeInt = 239
	apiTypeStr = "kOrientConstraint"
	typeIdInt = 1146049091
	MFnCls = om.MFnTransform
	pass

