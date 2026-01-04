

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Constraint = Catalogue.Constraint
else:
	from .. import retriever
	Constraint = retriever.getNodeCls("Constraint")
	assert Constraint

# add node doc



# region plug type defs
class ConstraintJointOrientXPlug(Plug):
	parent : ConstraintJointOrientPlug = PlugDescriptor("constraintJointOrient")
	node : ParentConstraint = None
	pass
class ConstraintJointOrientYPlug(Plug):
	parent : ConstraintJointOrientPlug = PlugDescriptor("constraintJointOrient")
	node : ParentConstraint = None
	pass
class ConstraintJointOrientZPlug(Plug):
	parent : ConstraintJointOrientPlug = PlugDescriptor("constraintJointOrient")
	node : ParentConstraint = None
	pass
class ConstraintJointOrientPlug(Plug):
	constraintJointOrientX_ : ConstraintJointOrientXPlug = PlugDescriptor("constraintJointOrientX")
	cjox_ : ConstraintJointOrientXPlug = PlugDescriptor("constraintJointOrientX")
	constraintJointOrientY_ : ConstraintJointOrientYPlug = PlugDescriptor("constraintJointOrientY")
	cjoy_ : ConstraintJointOrientYPlug = PlugDescriptor("constraintJointOrientY")
	constraintJointOrientZ_ : ConstraintJointOrientZPlug = PlugDescriptor("constraintJointOrientZ")
	cjoz_ : ConstraintJointOrientZPlug = PlugDescriptor("constraintJointOrientZ")
	node : ParentConstraint = None
	pass
class ConstraintParentInverseMatrixPlug(Plug):
	node : ParentConstraint = None
	pass
class ConstraintRotateXPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : ParentConstraint = None
	pass
class ConstraintRotateYPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : ParentConstraint = None
	pass
class ConstraintRotateZPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : ParentConstraint = None
	pass
class ConstraintRotatePlug(Plug):
	constraintRotateX_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	crx_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	constraintRotateY_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	cry_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	constraintRotateZ_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	crz_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	node : ParentConstraint = None
	pass
class ConstraintRotateOrderPlug(Plug):
	node : ParentConstraint = None
	pass
class ConstraintRotatePivotXPlug(Plug):
	parent : ConstraintRotatePivotPlug = PlugDescriptor("constraintRotatePivot")
	node : ParentConstraint = None
	pass
class ConstraintRotatePivotYPlug(Plug):
	parent : ConstraintRotatePivotPlug = PlugDescriptor("constraintRotatePivot")
	node : ParentConstraint = None
	pass
class ConstraintRotatePivotZPlug(Plug):
	parent : ConstraintRotatePivotPlug = PlugDescriptor("constraintRotatePivot")
	node : ParentConstraint = None
	pass
class ConstraintRotatePivotPlug(Plug):
	constraintRotatePivotX_ : ConstraintRotatePivotXPlug = PlugDescriptor("constraintRotatePivotX")
	crpx_ : ConstraintRotatePivotXPlug = PlugDescriptor("constraintRotatePivotX")
	constraintRotatePivotY_ : ConstraintRotatePivotYPlug = PlugDescriptor("constraintRotatePivotY")
	crpy_ : ConstraintRotatePivotYPlug = PlugDescriptor("constraintRotatePivotY")
	constraintRotatePivotZ_ : ConstraintRotatePivotZPlug = PlugDescriptor("constraintRotatePivotZ")
	crpz_ : ConstraintRotatePivotZPlug = PlugDescriptor("constraintRotatePivotZ")
	node : ParentConstraint = None
	pass
class ConstraintRotateTranslateXPlug(Plug):
	parent : ConstraintRotateTranslatePlug = PlugDescriptor("constraintRotateTranslate")
	node : ParentConstraint = None
	pass
class ConstraintRotateTranslateYPlug(Plug):
	parent : ConstraintRotateTranslatePlug = PlugDescriptor("constraintRotateTranslate")
	node : ParentConstraint = None
	pass
class ConstraintRotateTranslateZPlug(Plug):
	parent : ConstraintRotateTranslatePlug = PlugDescriptor("constraintRotateTranslate")
	node : ParentConstraint = None
	pass
class ConstraintRotateTranslatePlug(Plug):
	constraintRotateTranslateX_ : ConstraintRotateTranslateXPlug = PlugDescriptor("constraintRotateTranslateX")
	crtx_ : ConstraintRotateTranslateXPlug = PlugDescriptor("constraintRotateTranslateX")
	constraintRotateTranslateY_ : ConstraintRotateTranslateYPlug = PlugDescriptor("constraintRotateTranslateY")
	crty_ : ConstraintRotateTranslateYPlug = PlugDescriptor("constraintRotateTranslateY")
	constraintRotateTranslateZ_ : ConstraintRotateTranslateZPlug = PlugDescriptor("constraintRotateTranslateZ")
	crtz_ : ConstraintRotateTranslateZPlug = PlugDescriptor("constraintRotateTranslateZ")
	node : ParentConstraint = None
	pass
class ConstraintTranslateXPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : ParentConstraint = None
	pass
class ConstraintTranslateYPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : ParentConstraint = None
	pass
class ConstraintTranslateZPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : ParentConstraint = None
	pass
class ConstraintTranslatePlug(Plug):
	constraintTranslateX_ : ConstraintTranslateXPlug = PlugDescriptor("constraintTranslateX")
	ctx_ : ConstraintTranslateXPlug = PlugDescriptor("constraintTranslateX")
	constraintTranslateY_ : ConstraintTranslateYPlug = PlugDescriptor("constraintTranslateY")
	cty_ : ConstraintTranslateYPlug = PlugDescriptor("constraintTranslateY")
	constraintTranslateZ_ : ConstraintTranslateZPlug = PlugDescriptor("constraintTranslateZ")
	ctz_ : ConstraintTranslateZPlug = PlugDescriptor("constraintTranslateZ")
	node : ParentConstraint = None
	pass
class InterpCachePlug(Plug):
	node : ParentConstraint = None
	pass
class InterpTypePlug(Plug):
	node : ParentConstraint = None
	pass
class LastTargetRotateXPlug(Plug):
	parent : LastTargetRotatePlug = PlugDescriptor("lastTargetRotate")
	node : ParentConstraint = None
	pass
class LastTargetRotateYPlug(Plug):
	parent : LastTargetRotatePlug = PlugDescriptor("lastTargetRotate")
	node : ParentConstraint = None
	pass
class LastTargetRotateZPlug(Plug):
	parent : LastTargetRotatePlug = PlugDescriptor("lastTargetRotate")
	node : ParentConstraint = None
	pass
class LastTargetRotatePlug(Plug):
	lastTargetRotateX_ : LastTargetRotateXPlug = PlugDescriptor("lastTargetRotateX")
	lrx_ : LastTargetRotateXPlug = PlugDescriptor("lastTargetRotateX")
	lastTargetRotateY_ : LastTargetRotateYPlug = PlugDescriptor("lastTargetRotateY")
	lry_ : LastTargetRotateYPlug = PlugDescriptor("lastTargetRotateY")
	lastTargetRotateZ_ : LastTargetRotateZPlug = PlugDescriptor("lastTargetRotateZ")
	lrz_ : LastTargetRotateZPlug = PlugDescriptor("lastTargetRotateZ")
	node : ParentConstraint = None
	pass
class RestRotateXPlug(Plug):
	parent : RestRotatePlug = PlugDescriptor("restRotate")
	node : ParentConstraint = None
	pass
class RestRotateYPlug(Plug):
	parent : RestRotatePlug = PlugDescriptor("restRotate")
	node : ParentConstraint = None
	pass
class RestRotateZPlug(Plug):
	parent : RestRotatePlug = PlugDescriptor("restRotate")
	node : ParentConstraint = None
	pass
class RestRotatePlug(Plug):
	restRotateX_ : RestRotateXPlug = PlugDescriptor("restRotateX")
	rrx_ : RestRotateXPlug = PlugDescriptor("restRotateX")
	restRotateY_ : RestRotateYPlug = PlugDescriptor("restRotateY")
	rry_ : RestRotateYPlug = PlugDescriptor("restRotateY")
	restRotateZ_ : RestRotateZPlug = PlugDescriptor("restRotateZ")
	rrz_ : RestRotateZPlug = PlugDescriptor("restRotateZ")
	node : ParentConstraint = None
	pass
class RestTranslateXPlug(Plug):
	parent : RestTranslatePlug = PlugDescriptor("restTranslate")
	node : ParentConstraint = None
	pass
class RestTranslateYPlug(Plug):
	parent : RestTranslatePlug = PlugDescriptor("restTranslate")
	node : ParentConstraint = None
	pass
class RestTranslateZPlug(Plug):
	parent : RestTranslatePlug = PlugDescriptor("restTranslate")
	node : ParentConstraint = None
	pass
class RestTranslatePlug(Plug):
	restTranslateX_ : RestTranslateXPlug = PlugDescriptor("restTranslateX")
	rtx_ : RestTranslateXPlug = PlugDescriptor("restTranslateX")
	restTranslateY_ : RestTranslateYPlug = PlugDescriptor("restTranslateY")
	rty_ : RestTranslateYPlug = PlugDescriptor("restTranslateY")
	restTranslateZ_ : RestTranslateZPlug = PlugDescriptor("restTranslateZ")
	rtz_ : RestTranslateZPlug = PlugDescriptor("restTranslateZ")
	node : ParentConstraint = None
	pass
class RotationDecompositionTargetXPlug(Plug):
	parent : RotationDecompositionTargetPlug = PlugDescriptor("rotationDecompositionTarget")
	node : ParentConstraint = None
	pass
class RotationDecompositionTargetYPlug(Plug):
	parent : RotationDecompositionTargetPlug = PlugDescriptor("rotationDecompositionTarget")
	node : ParentConstraint = None
	pass
class RotationDecompositionTargetZPlug(Plug):
	parent : RotationDecompositionTargetPlug = PlugDescriptor("rotationDecompositionTarget")
	node : ParentConstraint = None
	pass
class RotationDecompositionTargetPlug(Plug):
	rotationDecompositionTargetX_ : RotationDecompositionTargetXPlug = PlugDescriptor("rotationDecompositionTargetX")
	rdtx_ : RotationDecompositionTargetXPlug = PlugDescriptor("rotationDecompositionTargetX")
	rotationDecompositionTargetY_ : RotationDecompositionTargetYPlug = PlugDescriptor("rotationDecompositionTargetY")
	rdty_ : RotationDecompositionTargetYPlug = PlugDescriptor("rotationDecompositionTargetY")
	rotationDecompositionTargetZ_ : RotationDecompositionTargetZPlug = PlugDescriptor("rotationDecompositionTargetZ")
	rdtz_ : RotationDecompositionTargetZPlug = PlugDescriptor("rotationDecompositionTargetZ")
	node : ParentConstraint = None
	pass
class TargetInverseScaleXPlug(Plug):
	parent : TargetInverseScalePlug = PlugDescriptor("targetInverseScale")
	node : ParentConstraint = None
	pass
class TargetInverseScaleYPlug(Plug):
	parent : TargetInverseScalePlug = PlugDescriptor("targetInverseScale")
	node : ParentConstraint = None
	pass
class TargetInverseScaleZPlug(Plug):
	parent : TargetInverseScalePlug = PlugDescriptor("targetInverseScale")
	node : ParentConstraint = None
	pass
class TargetInverseScalePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetInverseScaleX_ : TargetInverseScaleXPlug = PlugDescriptor("targetInverseScaleX")
	tisx_ : TargetInverseScaleXPlug = PlugDescriptor("targetInverseScaleX")
	targetInverseScaleY_ : TargetInverseScaleYPlug = PlugDescriptor("targetInverseScaleY")
	tisy_ : TargetInverseScaleYPlug = PlugDescriptor("targetInverseScaleY")
	targetInverseScaleZ_ : TargetInverseScaleZPlug = PlugDescriptor("targetInverseScaleZ")
	tisz_ : TargetInverseScaleZPlug = PlugDescriptor("targetInverseScaleZ")
	node : ParentConstraint = None
	pass
class TargetJointOrientXPlug(Plug):
	parent : TargetJointOrientPlug = PlugDescriptor("targetJointOrient")
	node : ParentConstraint = None
	pass
class TargetJointOrientYPlug(Plug):
	parent : TargetJointOrientPlug = PlugDescriptor("targetJointOrient")
	node : ParentConstraint = None
	pass
class TargetJointOrientZPlug(Plug):
	parent : TargetJointOrientPlug = PlugDescriptor("targetJointOrient")
	node : ParentConstraint = None
	pass
class TargetJointOrientPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetJointOrientX_ : TargetJointOrientXPlug = PlugDescriptor("targetJointOrientX")
	tjox_ : TargetJointOrientXPlug = PlugDescriptor("targetJointOrientX")
	targetJointOrientY_ : TargetJointOrientYPlug = PlugDescriptor("targetJointOrientY")
	tjoy_ : TargetJointOrientYPlug = PlugDescriptor("targetJointOrientY")
	targetJointOrientZ_ : TargetJointOrientZPlug = PlugDescriptor("targetJointOrientZ")
	tjoz_ : TargetJointOrientZPlug = PlugDescriptor("targetJointOrientZ")
	node : ParentConstraint = None
	pass
class TargetOffsetRotateXPlug(Plug):
	parent : TargetOffsetRotatePlug = PlugDescriptor("targetOffsetRotate")
	node : ParentConstraint = None
	pass
class TargetOffsetRotateYPlug(Plug):
	parent : TargetOffsetRotatePlug = PlugDescriptor("targetOffsetRotate")
	node : ParentConstraint = None
	pass
class TargetOffsetRotateZPlug(Plug):
	parent : TargetOffsetRotatePlug = PlugDescriptor("targetOffsetRotate")
	node : ParentConstraint = None
	pass
class TargetOffsetRotatePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetOffsetRotateX_ : TargetOffsetRotateXPlug = PlugDescriptor("targetOffsetRotateX")
	torx_ : TargetOffsetRotateXPlug = PlugDescriptor("targetOffsetRotateX")
	targetOffsetRotateY_ : TargetOffsetRotateYPlug = PlugDescriptor("targetOffsetRotateY")
	tory_ : TargetOffsetRotateYPlug = PlugDescriptor("targetOffsetRotateY")
	targetOffsetRotateZ_ : TargetOffsetRotateZPlug = PlugDescriptor("targetOffsetRotateZ")
	torz_ : TargetOffsetRotateZPlug = PlugDescriptor("targetOffsetRotateZ")
	node : ParentConstraint = None
	pass
class TargetOffsetTranslateXPlug(Plug):
	parent : TargetOffsetTranslatePlug = PlugDescriptor("targetOffsetTranslate")
	node : ParentConstraint = None
	pass
class TargetOffsetTranslateYPlug(Plug):
	parent : TargetOffsetTranslatePlug = PlugDescriptor("targetOffsetTranslate")
	node : ParentConstraint = None
	pass
class TargetOffsetTranslateZPlug(Plug):
	parent : TargetOffsetTranslatePlug = PlugDescriptor("targetOffsetTranslate")
	node : ParentConstraint = None
	pass
class TargetOffsetTranslatePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetOffsetTranslateX_ : TargetOffsetTranslateXPlug = PlugDescriptor("targetOffsetTranslateX")
	totx_ : TargetOffsetTranslateXPlug = PlugDescriptor("targetOffsetTranslateX")
	targetOffsetTranslateY_ : TargetOffsetTranslateYPlug = PlugDescriptor("targetOffsetTranslateY")
	toty_ : TargetOffsetTranslateYPlug = PlugDescriptor("targetOffsetTranslateY")
	targetOffsetTranslateZ_ : TargetOffsetTranslateZPlug = PlugDescriptor("targetOffsetTranslateZ")
	totz_ : TargetOffsetTranslateZPlug = PlugDescriptor("targetOffsetTranslateZ")
	node : ParentConstraint = None
	pass
class TargetParentMatrixPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : ParentConstraint = None
	pass
class TargetRotateXPlug(Plug):
	parent : TargetRotatePlug = PlugDescriptor("targetRotate")
	node : ParentConstraint = None
	pass
class TargetRotateYPlug(Plug):
	parent : TargetRotatePlug = PlugDescriptor("targetRotate")
	node : ParentConstraint = None
	pass
class TargetRotateZPlug(Plug):
	parent : TargetRotatePlug = PlugDescriptor("targetRotate")
	node : ParentConstraint = None
	pass
class TargetRotatePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetRotateX_ : TargetRotateXPlug = PlugDescriptor("targetRotateX")
	trx_ : TargetRotateXPlug = PlugDescriptor("targetRotateX")
	targetRotateY_ : TargetRotateYPlug = PlugDescriptor("targetRotateY")
	try_ : TargetRotateYPlug = PlugDescriptor("targetRotateY")
	targetRotateZ_ : TargetRotateZPlug = PlugDescriptor("targetRotateZ")
	trz_ : TargetRotateZPlug = PlugDescriptor("targetRotateZ")
	node : ParentConstraint = None
	pass
class TargetRotateCachedXPlug(Plug):
	parent : TargetRotateCachedPlug = PlugDescriptor("targetRotateCached")
	node : ParentConstraint = None
	pass
class TargetRotateCachedYPlug(Plug):
	parent : TargetRotateCachedPlug = PlugDescriptor("targetRotateCached")
	node : ParentConstraint = None
	pass
class TargetRotateCachedZPlug(Plug):
	parent : TargetRotateCachedPlug = PlugDescriptor("targetRotateCached")
	node : ParentConstraint = None
	pass
class TargetRotateCachedPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetRotateCachedX_ : TargetRotateCachedXPlug = PlugDescriptor("targetRotateCachedX")
	ctrx_ : TargetRotateCachedXPlug = PlugDescriptor("targetRotateCachedX")
	targetRotateCachedY_ : TargetRotateCachedYPlug = PlugDescriptor("targetRotateCachedY")
	ctry_ : TargetRotateCachedYPlug = PlugDescriptor("targetRotateCachedY")
	targetRotateCachedZ_ : TargetRotateCachedZPlug = PlugDescriptor("targetRotateCachedZ")
	ctrz_ : TargetRotateCachedZPlug = PlugDescriptor("targetRotateCachedZ")
	node : ParentConstraint = None
	pass
class TargetRotateOrderPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : ParentConstraint = None
	pass
class TargetRotatePivotXPlug(Plug):
	parent : TargetRotatePivotPlug = PlugDescriptor("targetRotatePivot")
	node : ParentConstraint = None
	pass
class TargetRotatePivotYPlug(Plug):
	parent : TargetRotatePivotPlug = PlugDescriptor("targetRotatePivot")
	node : ParentConstraint = None
	pass
class TargetRotatePivotZPlug(Plug):
	parent : TargetRotatePivotPlug = PlugDescriptor("targetRotatePivot")
	node : ParentConstraint = None
	pass
class TargetRotatePivotPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetRotatePivotX_ : TargetRotatePivotXPlug = PlugDescriptor("targetRotatePivotX")
	trpx_ : TargetRotatePivotXPlug = PlugDescriptor("targetRotatePivotX")
	targetRotatePivotY_ : TargetRotatePivotYPlug = PlugDescriptor("targetRotatePivotY")
	trpy_ : TargetRotatePivotYPlug = PlugDescriptor("targetRotatePivotY")
	targetRotatePivotZ_ : TargetRotatePivotZPlug = PlugDescriptor("targetRotatePivotZ")
	trpz_ : TargetRotatePivotZPlug = PlugDescriptor("targetRotatePivotZ")
	node : ParentConstraint = None
	pass
class TargetRotateTranslateXPlug(Plug):
	parent : TargetRotateTranslatePlug = PlugDescriptor("targetRotateTranslate")
	node : ParentConstraint = None
	pass
class TargetRotateTranslateYPlug(Plug):
	parent : TargetRotateTranslatePlug = PlugDescriptor("targetRotateTranslate")
	node : ParentConstraint = None
	pass
class TargetRotateTranslateZPlug(Plug):
	parent : TargetRotateTranslatePlug = PlugDescriptor("targetRotateTranslate")
	node : ParentConstraint = None
	pass
class TargetRotateTranslatePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetRotateTranslateX_ : TargetRotateTranslateXPlug = PlugDescriptor("targetRotateTranslateX")
	trtx_ : TargetRotateTranslateXPlug = PlugDescriptor("targetRotateTranslateX")
	targetRotateTranslateY_ : TargetRotateTranslateYPlug = PlugDescriptor("targetRotateTranslateY")
	trty_ : TargetRotateTranslateYPlug = PlugDescriptor("targetRotateTranslateY")
	targetRotateTranslateZ_ : TargetRotateTranslateZPlug = PlugDescriptor("targetRotateTranslateZ")
	trtz_ : TargetRotateTranslateZPlug = PlugDescriptor("targetRotateTranslateZ")
	node : ParentConstraint = None
	pass
class TargetScaleXPlug(Plug):
	parent : TargetScalePlug = PlugDescriptor("targetScale")
	node : ParentConstraint = None
	pass
class TargetScaleYPlug(Plug):
	parent : TargetScalePlug = PlugDescriptor("targetScale")
	node : ParentConstraint = None
	pass
class TargetScaleZPlug(Plug):
	parent : TargetScalePlug = PlugDescriptor("targetScale")
	node : ParentConstraint = None
	pass
class TargetScalePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetScaleX_ : TargetScaleXPlug = PlugDescriptor("targetScaleX")
	tsx_ : TargetScaleXPlug = PlugDescriptor("targetScaleX")
	targetScaleY_ : TargetScaleYPlug = PlugDescriptor("targetScaleY")
	tsy_ : TargetScaleYPlug = PlugDescriptor("targetScaleY")
	targetScaleZ_ : TargetScaleZPlug = PlugDescriptor("targetScaleZ")
	tsz_ : TargetScaleZPlug = PlugDescriptor("targetScaleZ")
	node : ParentConstraint = None
	pass
class TargetScaleCompensatePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : ParentConstraint = None
	pass
class TargetTranslateXPlug(Plug):
	parent : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	node : ParentConstraint = None
	pass
class TargetTranslateYPlug(Plug):
	parent : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	node : ParentConstraint = None
	pass
class TargetTranslateZPlug(Plug):
	parent : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	node : ParentConstraint = None
	pass
class TargetTranslatePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetTranslateX_ : TargetTranslateXPlug = PlugDescriptor("targetTranslateX")
	ttx_ : TargetTranslateXPlug = PlugDescriptor("targetTranslateX")
	targetTranslateY_ : TargetTranslateYPlug = PlugDescriptor("targetTranslateY")
	tty_ : TargetTranslateYPlug = PlugDescriptor("targetTranslateY")
	targetTranslateZ_ : TargetTranslateZPlug = PlugDescriptor("targetTranslateZ")
	ttz_ : TargetTranslateZPlug = PlugDescriptor("targetTranslateZ")
	node : ParentConstraint = None
	pass
class TargetWeightPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : ParentConstraint = None
	pass
class TargetPlug(Plug):
	targetInverseScale_ : TargetInverseScalePlug = PlugDescriptor("targetInverseScale")
	tis_ : TargetInverseScalePlug = PlugDescriptor("targetInverseScale")
	targetJointOrient_ : TargetJointOrientPlug = PlugDescriptor("targetJointOrient")
	tjo_ : TargetJointOrientPlug = PlugDescriptor("targetJointOrient")
	targetOffsetRotate_ : TargetOffsetRotatePlug = PlugDescriptor("targetOffsetRotate")
	tor_ : TargetOffsetRotatePlug = PlugDescriptor("targetOffsetRotate")
	targetOffsetTranslate_ : TargetOffsetTranslatePlug = PlugDescriptor("targetOffsetTranslate")
	tot_ : TargetOffsetTranslatePlug = PlugDescriptor("targetOffsetTranslate")
	targetParentMatrix_ : TargetParentMatrixPlug = PlugDescriptor("targetParentMatrix")
	tpm_ : TargetParentMatrixPlug = PlugDescriptor("targetParentMatrix")
	targetRotate_ : TargetRotatePlug = PlugDescriptor("targetRotate")
	tr_ : TargetRotatePlug = PlugDescriptor("targetRotate")
	targetRotateCached_ : TargetRotateCachedPlug = PlugDescriptor("targetRotateCached")
	ctr_ : TargetRotateCachedPlug = PlugDescriptor("targetRotateCached")
	targetRotateOrder_ : TargetRotateOrderPlug = PlugDescriptor("targetRotateOrder")
	tro_ : TargetRotateOrderPlug = PlugDescriptor("targetRotateOrder")
	targetRotatePivot_ : TargetRotatePivotPlug = PlugDescriptor("targetRotatePivot")
	trp_ : TargetRotatePivotPlug = PlugDescriptor("targetRotatePivot")
	targetRotateTranslate_ : TargetRotateTranslatePlug = PlugDescriptor("targetRotateTranslate")
	trt_ : TargetRotateTranslatePlug = PlugDescriptor("targetRotateTranslate")
	targetScale_ : TargetScalePlug = PlugDescriptor("targetScale")
	ts_ : TargetScalePlug = PlugDescriptor("targetScale")
	targetScaleCompensate_ : TargetScaleCompensatePlug = PlugDescriptor("targetScaleCompensate")
	tsc_ : TargetScaleCompensatePlug = PlugDescriptor("targetScaleCompensate")
	targetTranslate_ : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	tt_ : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	targetWeight_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	tw_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	node : ParentConstraint = None
	pass
class UseDecompositionTargetPlug(Plug):
	node : ParentConstraint = None
	pass
# endregion


# define node class
class ParentConstraint(Constraint):
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
	interpCache_ : InterpCachePlug = PlugDescriptor("interpCache")
	interpType_ : InterpTypePlug = PlugDescriptor("interpType")
	lastTargetRotateX_ : LastTargetRotateXPlug = PlugDescriptor("lastTargetRotateX")
	lastTargetRotateY_ : LastTargetRotateYPlug = PlugDescriptor("lastTargetRotateY")
	lastTargetRotateZ_ : LastTargetRotateZPlug = PlugDescriptor("lastTargetRotateZ")
	lastTargetRotate_ : LastTargetRotatePlug = PlugDescriptor("lastTargetRotate")
	restRotateX_ : RestRotateXPlug = PlugDescriptor("restRotateX")
	restRotateY_ : RestRotateYPlug = PlugDescriptor("restRotateY")
	restRotateZ_ : RestRotateZPlug = PlugDescriptor("restRotateZ")
	restRotate_ : RestRotatePlug = PlugDescriptor("restRotate")
	restTranslateX_ : RestTranslateXPlug = PlugDescriptor("restTranslateX")
	restTranslateY_ : RestTranslateYPlug = PlugDescriptor("restTranslateY")
	restTranslateZ_ : RestTranslateZPlug = PlugDescriptor("restTranslateZ")
	restTranslate_ : RestTranslatePlug = PlugDescriptor("restTranslate")
	rotationDecompositionTargetX_ : RotationDecompositionTargetXPlug = PlugDescriptor("rotationDecompositionTargetX")
	rotationDecompositionTargetY_ : RotationDecompositionTargetYPlug = PlugDescriptor("rotationDecompositionTargetY")
	rotationDecompositionTargetZ_ : RotationDecompositionTargetZPlug = PlugDescriptor("rotationDecompositionTargetZ")
	rotationDecompositionTarget_ : RotationDecompositionTargetPlug = PlugDescriptor("rotationDecompositionTarget")
	targetInverseScaleX_ : TargetInverseScaleXPlug = PlugDescriptor("targetInverseScaleX")
	targetInverseScaleY_ : TargetInverseScaleYPlug = PlugDescriptor("targetInverseScaleY")
	targetInverseScaleZ_ : TargetInverseScaleZPlug = PlugDescriptor("targetInverseScaleZ")
	targetInverseScale_ : TargetInverseScalePlug = PlugDescriptor("targetInverseScale")
	targetJointOrientX_ : TargetJointOrientXPlug = PlugDescriptor("targetJointOrientX")
	targetJointOrientY_ : TargetJointOrientYPlug = PlugDescriptor("targetJointOrientY")
	targetJointOrientZ_ : TargetJointOrientZPlug = PlugDescriptor("targetJointOrientZ")
	targetJointOrient_ : TargetJointOrientPlug = PlugDescriptor("targetJointOrient")
	targetOffsetRotateX_ : TargetOffsetRotateXPlug = PlugDescriptor("targetOffsetRotateX")
	targetOffsetRotateY_ : TargetOffsetRotateYPlug = PlugDescriptor("targetOffsetRotateY")
	targetOffsetRotateZ_ : TargetOffsetRotateZPlug = PlugDescriptor("targetOffsetRotateZ")
	targetOffsetRotate_ : TargetOffsetRotatePlug = PlugDescriptor("targetOffsetRotate")
	targetOffsetTranslateX_ : TargetOffsetTranslateXPlug = PlugDescriptor("targetOffsetTranslateX")
	targetOffsetTranslateY_ : TargetOffsetTranslateYPlug = PlugDescriptor("targetOffsetTranslateY")
	targetOffsetTranslateZ_ : TargetOffsetTranslateZPlug = PlugDescriptor("targetOffsetTranslateZ")
	targetOffsetTranslate_ : TargetOffsetTranslatePlug = PlugDescriptor("targetOffsetTranslate")
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
	targetRotatePivotX_ : TargetRotatePivotXPlug = PlugDescriptor("targetRotatePivotX")
	targetRotatePivotY_ : TargetRotatePivotYPlug = PlugDescriptor("targetRotatePivotY")
	targetRotatePivotZ_ : TargetRotatePivotZPlug = PlugDescriptor("targetRotatePivotZ")
	targetRotatePivot_ : TargetRotatePivotPlug = PlugDescriptor("targetRotatePivot")
	targetRotateTranslateX_ : TargetRotateTranslateXPlug = PlugDescriptor("targetRotateTranslateX")
	targetRotateTranslateY_ : TargetRotateTranslateYPlug = PlugDescriptor("targetRotateTranslateY")
	targetRotateTranslateZ_ : TargetRotateTranslateZPlug = PlugDescriptor("targetRotateTranslateZ")
	targetRotateTranslate_ : TargetRotateTranslatePlug = PlugDescriptor("targetRotateTranslate")
	targetScaleX_ : TargetScaleXPlug = PlugDescriptor("targetScaleX")
	targetScaleY_ : TargetScaleYPlug = PlugDescriptor("targetScaleY")
	targetScaleZ_ : TargetScaleZPlug = PlugDescriptor("targetScaleZ")
	targetScale_ : TargetScalePlug = PlugDescriptor("targetScale")
	targetScaleCompensate_ : TargetScaleCompensatePlug = PlugDescriptor("targetScaleCompensate")
	targetTranslateX_ : TargetTranslateXPlug = PlugDescriptor("targetTranslateX")
	targetTranslateY_ : TargetTranslateYPlug = PlugDescriptor("targetTranslateY")
	targetTranslateZ_ : TargetTranslateZPlug = PlugDescriptor("targetTranslateZ")
	targetTranslate_ : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	targetWeight_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	target_ : TargetPlug = PlugDescriptor("target")
	useDecompositionTarget_ : UseDecompositionTargetPlug = PlugDescriptor("useDecompositionTarget")

	# node attributes

	typeName = "parentConstraint"
	apiTypeInt = 242
	apiTypeStr = "kParentConstraint"
	typeIdInt = 1146110290
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["constraintJointOrientX", "constraintJointOrientY", "constraintJointOrientZ", "constraintJointOrient", "constraintParentInverseMatrix", "constraintRotateX", "constraintRotateY", "constraintRotateZ", "constraintRotate", "constraintRotateOrder", "constraintRotatePivotX", "constraintRotatePivotY", "constraintRotatePivotZ", "constraintRotatePivot", "constraintRotateTranslateX", "constraintRotateTranslateY", "constraintRotateTranslateZ", "constraintRotateTranslate", "constraintTranslateX", "constraintTranslateY", "constraintTranslateZ", "constraintTranslate", "interpCache", "interpType", "lastTargetRotateX", "lastTargetRotateY", "lastTargetRotateZ", "lastTargetRotate", "restRotateX", "restRotateY", "restRotateZ", "restRotate", "restTranslateX", "restTranslateY", "restTranslateZ", "restTranslate", "rotationDecompositionTargetX", "rotationDecompositionTargetY", "rotationDecompositionTargetZ", "rotationDecompositionTarget", "targetInverseScaleX", "targetInverseScaleY", "targetInverseScaleZ", "targetInverseScale", "targetJointOrientX", "targetJointOrientY", "targetJointOrientZ", "targetJointOrient", "targetOffsetRotateX", "targetOffsetRotateY", "targetOffsetRotateZ", "targetOffsetRotate", "targetOffsetTranslateX", "targetOffsetTranslateY", "targetOffsetTranslateZ", "targetOffsetTranslate", "targetParentMatrix", "targetRotateX", "targetRotateY", "targetRotateZ", "targetRotate", "targetRotateCachedX", "targetRotateCachedY", "targetRotateCachedZ", "targetRotateCached", "targetRotateOrder", "targetRotatePivotX", "targetRotatePivotY", "targetRotatePivotZ", "targetRotatePivot", "targetRotateTranslateX", "targetRotateTranslateY", "targetRotateTranslateZ", "targetRotateTranslate", "targetScaleX", "targetScaleY", "targetScaleZ", "targetScale", "targetScaleCompensate", "targetTranslateX", "targetTranslateY", "targetTranslateZ", "targetTranslate", "targetWeight", "target", "useDecompositionTarget"]
	nodeLeafPlugs = ["constraintJointOrient", "constraintParentInverseMatrix", "constraintRotate", "constraintRotateOrder", "constraintRotatePivot", "constraintRotateTranslate", "constraintTranslate", "interpCache", "interpType", "lastTargetRotate", "restRotate", "restTranslate", "rotationDecompositionTarget", "target", "useDecompositionTarget"]
	pass

