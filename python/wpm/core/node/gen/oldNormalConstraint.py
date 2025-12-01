

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
	node : OldNormalConstraint = None
	pass
class AimVectorYPlug(Plug):
	parent : AimVectorPlug = PlugDescriptor("aimVector")
	node : OldNormalConstraint = None
	pass
class AimVectorZPlug(Plug):
	parent : AimVectorPlug = PlugDescriptor("aimVector")
	node : OldNormalConstraint = None
	pass
class AimVectorPlug(Plug):
	aimVectorX_ : AimVectorXPlug = PlugDescriptor("aimVectorX")
	ax_ : AimVectorXPlug = PlugDescriptor("aimVectorX")
	aimVectorY_ : AimVectorYPlug = PlugDescriptor("aimVectorY")
	ay_ : AimVectorYPlug = PlugDescriptor("aimVectorY")
	aimVectorZ_ : AimVectorZPlug = PlugDescriptor("aimVectorZ")
	az_ : AimVectorZPlug = PlugDescriptor("aimVectorZ")
	node : OldNormalConstraint = None
	pass
class ConstraintParentInverseMatrixPlug(Plug):
	node : OldNormalConstraint = None
	pass
class ConstraintRotateXPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : OldNormalConstraint = None
	pass
class ConstraintRotateYPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : OldNormalConstraint = None
	pass
class ConstraintRotateZPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : OldNormalConstraint = None
	pass
class ConstraintRotatePlug(Plug):
	constraintRotateX_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	crx_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	constraintRotateY_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	cry_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	constraintRotateZ_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	crz_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	node : OldNormalConstraint = None
	pass
class TargetParentMatrixPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : OldNormalConstraint = None
	pass
class TargetRotatePivotXPlug(Plug):
	parent : TargetRotatePivotPlug = PlugDescriptor("targetRotatePivot")
	node : OldNormalConstraint = None
	pass
class TargetRotatePivotYPlug(Plug):
	parent : TargetRotatePivotPlug = PlugDescriptor("targetRotatePivot")
	node : OldNormalConstraint = None
	pass
class TargetRotatePivotZPlug(Plug):
	parent : TargetRotatePivotPlug = PlugDescriptor("targetRotatePivot")
	node : OldNormalConstraint = None
	pass
class TargetRotatePivotPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetRotatePivotX_ : TargetRotatePivotXPlug = PlugDescriptor("targetRotatePivotX")
	trpx_ : TargetRotatePivotXPlug = PlugDescriptor("targetRotatePivotX")
	targetRotatePivotY_ : TargetRotatePivotYPlug = PlugDescriptor("targetRotatePivotY")
	trpy_ : TargetRotatePivotYPlug = PlugDescriptor("targetRotatePivotY")
	targetRotatePivotZ_ : TargetRotatePivotZPlug = PlugDescriptor("targetRotatePivotZ")
	trpz_ : TargetRotatePivotZPlug = PlugDescriptor("targetRotatePivotZ")
	node : OldNormalConstraint = None
	pass
class TargetRotateTranslateXPlug(Plug):
	parent : TargetRotateTranslatePlug = PlugDescriptor("targetRotateTranslate")
	node : OldNormalConstraint = None
	pass
class TargetRotateTranslateYPlug(Plug):
	parent : TargetRotateTranslatePlug = PlugDescriptor("targetRotateTranslate")
	node : OldNormalConstraint = None
	pass
class TargetRotateTranslateZPlug(Plug):
	parent : TargetRotateTranslatePlug = PlugDescriptor("targetRotateTranslate")
	node : OldNormalConstraint = None
	pass
class TargetRotateTranslatePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetRotateTranslateX_ : TargetRotateTranslateXPlug = PlugDescriptor("targetRotateTranslateX")
	trtx_ : TargetRotateTranslateXPlug = PlugDescriptor("targetRotateTranslateX")
	targetRotateTranslateY_ : TargetRotateTranslateYPlug = PlugDescriptor("targetRotateTranslateY")
	trty_ : TargetRotateTranslateYPlug = PlugDescriptor("targetRotateTranslateY")
	targetRotateTranslateZ_ : TargetRotateTranslateZPlug = PlugDescriptor("targetRotateTranslateZ")
	trtz_ : TargetRotateTranslateZPlug = PlugDescriptor("targetRotateTranslateZ")
	node : OldNormalConstraint = None
	pass
class TargetTranslateXPlug(Plug):
	parent : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	node : OldNormalConstraint = None
	pass
class TargetTranslateYPlug(Plug):
	parent : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	node : OldNormalConstraint = None
	pass
class TargetTranslateZPlug(Plug):
	parent : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	node : OldNormalConstraint = None
	pass
class TargetTranslatePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetTranslateX_ : TargetTranslateXPlug = PlugDescriptor("targetTranslateX")
	ttx_ : TargetTranslateXPlug = PlugDescriptor("targetTranslateX")
	targetTranslateY_ : TargetTranslateYPlug = PlugDescriptor("targetTranslateY")
	tty_ : TargetTranslateYPlug = PlugDescriptor("targetTranslateY")
	targetTranslateZ_ : TargetTranslateZPlug = PlugDescriptor("targetTranslateZ")
	ttz_ : TargetTranslateZPlug = PlugDescriptor("targetTranslateZ")
	node : OldNormalConstraint = None
	pass
class TargetWeightPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : OldNormalConstraint = None
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
	node : OldNormalConstraint = None
	pass
class UpVectorXPlug(Plug):
	parent : UpVectorPlug = PlugDescriptor("upVector")
	node : OldNormalConstraint = None
	pass
class UpVectorYPlug(Plug):
	parent : UpVectorPlug = PlugDescriptor("upVector")
	node : OldNormalConstraint = None
	pass
class UpVectorZPlug(Plug):
	parent : UpVectorPlug = PlugDescriptor("upVector")
	node : OldNormalConstraint = None
	pass
class UpVectorPlug(Plug):
	upVectorX_ : UpVectorXPlug = PlugDescriptor("upVectorX")
	ux_ : UpVectorXPlug = PlugDescriptor("upVectorX")
	upVectorY_ : UpVectorYPlug = PlugDescriptor("upVectorY")
	uy_ : UpVectorYPlug = PlugDescriptor("upVectorY")
	upVectorZ_ : UpVectorZPlug = PlugDescriptor("upVectorZ")
	uz_ : UpVectorZPlug = PlugDescriptor("upVectorZ")
	node : OldNormalConstraint = None
	pass
class WorldUpVectorXPlug(Plug):
	parent : WorldUpVectorPlug = PlugDescriptor("worldUpVector")
	node : OldNormalConstraint = None
	pass
class WorldUpVectorYPlug(Plug):
	parent : WorldUpVectorPlug = PlugDescriptor("worldUpVector")
	node : OldNormalConstraint = None
	pass
class WorldUpVectorZPlug(Plug):
	parent : WorldUpVectorPlug = PlugDescriptor("worldUpVector")
	node : OldNormalConstraint = None
	pass
class WorldUpVectorPlug(Plug):
	worldUpVectorX_ : WorldUpVectorXPlug = PlugDescriptor("worldUpVectorX")
	wux_ : WorldUpVectorXPlug = PlugDescriptor("worldUpVectorX")
	worldUpVectorY_ : WorldUpVectorYPlug = PlugDescriptor("worldUpVectorY")
	wuy_ : WorldUpVectorYPlug = PlugDescriptor("worldUpVectorY")
	worldUpVectorZ_ : WorldUpVectorZPlug = PlugDescriptor("worldUpVectorZ")
	wuz_ : WorldUpVectorZPlug = PlugDescriptor("worldUpVectorZ")
	node : OldNormalConstraint = None
	pass
# endregion


# define node class
class OldNormalConstraint(Constraint):
	aimVectorX_ : AimVectorXPlug = PlugDescriptor("aimVectorX")
	aimVectorY_ : AimVectorYPlug = PlugDescriptor("aimVectorY")
	aimVectorZ_ : AimVectorZPlug = PlugDescriptor("aimVectorZ")
	aimVector_ : AimVectorPlug = PlugDescriptor("aimVector")
	constraintParentInverseMatrix_ : ConstraintParentInverseMatrixPlug = PlugDescriptor("constraintParentInverseMatrix")
	constraintRotateX_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	constraintRotateY_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	constraintRotateZ_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	constraintRotate_ : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
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
	worldUpVectorX_ : WorldUpVectorXPlug = PlugDescriptor("worldUpVectorX")
	worldUpVectorY_ : WorldUpVectorYPlug = PlugDescriptor("worldUpVectorY")
	worldUpVectorZ_ : WorldUpVectorZPlug = PlugDescriptor("worldUpVectorZ")
	worldUpVector_ : WorldUpVectorPlug = PlugDescriptor("worldUpVector")

	# node attributes

	typeName = "oldNormalConstraint"
	typeIdInt = 1145983555
	nodeLeafClassAttrs = ["aimVectorX", "aimVectorY", "aimVectorZ", "aimVector", "constraintParentInverseMatrix", "constraintRotateX", "constraintRotateY", "constraintRotateZ", "constraintRotate", "targetParentMatrix", "targetRotatePivotX", "targetRotatePivotY", "targetRotatePivotZ", "targetRotatePivot", "targetRotateTranslateX", "targetRotateTranslateY", "targetRotateTranslateZ", "targetRotateTranslate", "targetTranslateX", "targetTranslateY", "targetTranslateZ", "targetTranslate", "targetWeight", "target", "upVectorX", "upVectorY", "upVectorZ", "upVector", "worldUpVectorX", "worldUpVectorY", "worldUpVectorZ", "worldUpVector"]
	nodeLeafPlugs = ["aimVector", "constraintParentInverseMatrix", "constraintRotate", "target", "upVector", "worldUpVector"]
	pass

