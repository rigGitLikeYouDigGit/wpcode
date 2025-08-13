

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
class ConstraintOffsetPolarityPlug(Plug):
	node : PointConstraint = None
	pass
class ConstraintParentInverseMatrixPlug(Plug):
	node : PointConstraint = None
	pass
class ConstraintRotatePivotXPlug(Plug):
	parent : ConstraintRotatePivotPlug = PlugDescriptor("constraintRotatePivot")
	node : PointConstraint = None
	pass
class ConstraintRotatePivotYPlug(Plug):
	parent : ConstraintRotatePivotPlug = PlugDescriptor("constraintRotatePivot")
	node : PointConstraint = None
	pass
class ConstraintRotatePivotZPlug(Plug):
	parent : ConstraintRotatePivotPlug = PlugDescriptor("constraintRotatePivot")
	node : PointConstraint = None
	pass
class ConstraintRotatePivotPlug(Plug):
	constraintRotatePivotX_ : ConstraintRotatePivotXPlug = PlugDescriptor("constraintRotatePivotX")
	crpx_ : ConstraintRotatePivotXPlug = PlugDescriptor("constraintRotatePivotX")
	constraintRotatePivotY_ : ConstraintRotatePivotYPlug = PlugDescriptor("constraintRotatePivotY")
	crpy_ : ConstraintRotatePivotYPlug = PlugDescriptor("constraintRotatePivotY")
	constraintRotatePivotZ_ : ConstraintRotatePivotZPlug = PlugDescriptor("constraintRotatePivotZ")
	crpz_ : ConstraintRotatePivotZPlug = PlugDescriptor("constraintRotatePivotZ")
	node : PointConstraint = None
	pass
class ConstraintRotateTranslateXPlug(Plug):
	parent : ConstraintRotateTranslatePlug = PlugDescriptor("constraintRotateTranslate")
	node : PointConstraint = None
	pass
class ConstraintRotateTranslateYPlug(Plug):
	parent : ConstraintRotateTranslatePlug = PlugDescriptor("constraintRotateTranslate")
	node : PointConstraint = None
	pass
class ConstraintRotateTranslateZPlug(Plug):
	parent : ConstraintRotateTranslatePlug = PlugDescriptor("constraintRotateTranslate")
	node : PointConstraint = None
	pass
class ConstraintRotateTranslatePlug(Plug):
	constraintRotateTranslateX_ : ConstraintRotateTranslateXPlug = PlugDescriptor("constraintRotateTranslateX")
	crtx_ : ConstraintRotateTranslateXPlug = PlugDescriptor("constraintRotateTranslateX")
	constraintRotateTranslateY_ : ConstraintRotateTranslateYPlug = PlugDescriptor("constraintRotateTranslateY")
	crty_ : ConstraintRotateTranslateYPlug = PlugDescriptor("constraintRotateTranslateY")
	constraintRotateTranslateZ_ : ConstraintRotateTranslateZPlug = PlugDescriptor("constraintRotateTranslateZ")
	crtz_ : ConstraintRotateTranslateZPlug = PlugDescriptor("constraintRotateTranslateZ")
	node : PointConstraint = None
	pass
class ConstraintTranslateXPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : PointConstraint = None
	pass
class ConstraintTranslateYPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : PointConstraint = None
	pass
class ConstraintTranslateZPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : PointConstraint = None
	pass
class ConstraintTranslatePlug(Plug):
	constraintTranslateX_ : ConstraintTranslateXPlug = PlugDescriptor("constraintTranslateX")
	ctx_ : ConstraintTranslateXPlug = PlugDescriptor("constraintTranslateX")
	constraintTranslateY_ : ConstraintTranslateYPlug = PlugDescriptor("constraintTranslateY")
	cty_ : ConstraintTranslateYPlug = PlugDescriptor("constraintTranslateY")
	constraintTranslateZ_ : ConstraintTranslateZPlug = PlugDescriptor("constraintTranslateZ")
	ctz_ : ConstraintTranslateZPlug = PlugDescriptor("constraintTranslateZ")
	node : PointConstraint = None
	pass
class OffsetXPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : PointConstraint = None
	pass
class OffsetYPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : PointConstraint = None
	pass
class OffsetZPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : PointConstraint = None
	pass
class OffsetPlug(Plug):
	offsetX_ : OffsetXPlug = PlugDescriptor("offsetX")
	ox_ : OffsetXPlug = PlugDescriptor("offsetX")
	offsetY_ : OffsetYPlug = PlugDescriptor("offsetY")
	oy_ : OffsetYPlug = PlugDescriptor("offsetY")
	offsetZ_ : OffsetZPlug = PlugDescriptor("offsetZ")
	oz_ : OffsetZPlug = PlugDescriptor("offsetZ")
	node : PointConstraint = None
	pass
class RestTranslateXPlug(Plug):
	parent : RestTranslatePlug = PlugDescriptor("restTranslate")
	node : PointConstraint = None
	pass
class RestTranslateYPlug(Plug):
	parent : RestTranslatePlug = PlugDescriptor("restTranslate")
	node : PointConstraint = None
	pass
class RestTranslateZPlug(Plug):
	parent : RestTranslatePlug = PlugDescriptor("restTranslate")
	node : PointConstraint = None
	pass
class RestTranslatePlug(Plug):
	restTranslateX_ : RestTranslateXPlug = PlugDescriptor("restTranslateX")
	rtx_ : RestTranslateXPlug = PlugDescriptor("restTranslateX")
	restTranslateY_ : RestTranslateYPlug = PlugDescriptor("restTranslateY")
	rty_ : RestTranslateYPlug = PlugDescriptor("restTranslateY")
	restTranslateZ_ : RestTranslateZPlug = PlugDescriptor("restTranslateZ")
	rtz_ : RestTranslateZPlug = PlugDescriptor("restTranslateZ")
	node : PointConstraint = None
	pass
class TargetParentMatrixPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : PointConstraint = None
	pass
class TargetRotatePivotXPlug(Plug):
	parent : TargetRotatePivotPlug = PlugDescriptor("targetRotatePivot")
	node : PointConstraint = None
	pass
class TargetRotatePivotYPlug(Plug):
	parent : TargetRotatePivotPlug = PlugDescriptor("targetRotatePivot")
	node : PointConstraint = None
	pass
class TargetRotatePivotZPlug(Plug):
	parent : TargetRotatePivotPlug = PlugDescriptor("targetRotatePivot")
	node : PointConstraint = None
	pass
class TargetRotatePivotPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetRotatePivotX_ : TargetRotatePivotXPlug = PlugDescriptor("targetRotatePivotX")
	trpx_ : TargetRotatePivotXPlug = PlugDescriptor("targetRotatePivotX")
	targetRotatePivotY_ : TargetRotatePivotYPlug = PlugDescriptor("targetRotatePivotY")
	trpy_ : TargetRotatePivotYPlug = PlugDescriptor("targetRotatePivotY")
	targetRotatePivotZ_ : TargetRotatePivotZPlug = PlugDescriptor("targetRotatePivotZ")
	trpz_ : TargetRotatePivotZPlug = PlugDescriptor("targetRotatePivotZ")
	node : PointConstraint = None
	pass
class TargetRotateTranslateXPlug(Plug):
	parent : TargetRotateTranslatePlug = PlugDescriptor("targetRotateTranslate")
	node : PointConstraint = None
	pass
class TargetRotateTranslateYPlug(Plug):
	parent : TargetRotateTranslatePlug = PlugDescriptor("targetRotateTranslate")
	node : PointConstraint = None
	pass
class TargetRotateTranslateZPlug(Plug):
	parent : TargetRotateTranslatePlug = PlugDescriptor("targetRotateTranslate")
	node : PointConstraint = None
	pass
class TargetRotateTranslatePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetRotateTranslateX_ : TargetRotateTranslateXPlug = PlugDescriptor("targetRotateTranslateX")
	trtx_ : TargetRotateTranslateXPlug = PlugDescriptor("targetRotateTranslateX")
	targetRotateTranslateY_ : TargetRotateTranslateYPlug = PlugDescriptor("targetRotateTranslateY")
	trty_ : TargetRotateTranslateYPlug = PlugDescriptor("targetRotateTranslateY")
	targetRotateTranslateZ_ : TargetRotateTranslateZPlug = PlugDescriptor("targetRotateTranslateZ")
	trtz_ : TargetRotateTranslateZPlug = PlugDescriptor("targetRotateTranslateZ")
	node : PointConstraint = None
	pass
class TargetTranslateXPlug(Plug):
	parent : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	node : PointConstraint = None
	pass
class TargetTranslateYPlug(Plug):
	parent : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	node : PointConstraint = None
	pass
class TargetTranslateZPlug(Plug):
	parent : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	node : PointConstraint = None
	pass
class TargetTranslatePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetTranslateX_ : TargetTranslateXPlug = PlugDescriptor("targetTranslateX")
	ttx_ : TargetTranslateXPlug = PlugDescriptor("targetTranslateX")
	targetTranslateY_ : TargetTranslateYPlug = PlugDescriptor("targetTranslateY")
	tty_ : TargetTranslateYPlug = PlugDescriptor("targetTranslateY")
	targetTranslateZ_ : TargetTranslateZPlug = PlugDescriptor("targetTranslateZ")
	ttz_ : TargetTranslateZPlug = PlugDescriptor("targetTranslateZ")
	node : PointConstraint = None
	pass
class TargetWeightPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : PointConstraint = None
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
	node : PointConstraint = None
	pass
# endregion


# define node class
class PointConstraint(Constraint):
	constraintOffsetPolarity_ : ConstraintOffsetPolarityPlug = PlugDescriptor("constraintOffsetPolarity")
	constraintParentInverseMatrix_ : ConstraintParentInverseMatrixPlug = PlugDescriptor("constraintParentInverseMatrix")
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
	offsetX_ : OffsetXPlug = PlugDescriptor("offsetX")
	offsetY_ : OffsetYPlug = PlugDescriptor("offsetY")
	offsetZ_ : OffsetZPlug = PlugDescriptor("offsetZ")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	restTranslateX_ : RestTranslateXPlug = PlugDescriptor("restTranslateX")
	restTranslateY_ : RestTranslateYPlug = PlugDescriptor("restTranslateY")
	restTranslateZ_ : RestTranslateZPlug = PlugDescriptor("restTranslateZ")
	restTranslate_ : RestTranslatePlug = PlugDescriptor("restTranslate")
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

	# node attributes

	typeName = "pointConstraint"
	apiTypeInt = 240
	apiTypeStr = "kPointConstraint"
	typeIdInt = 1146115139
	MFnCls = om.MFnTransform
	pass

