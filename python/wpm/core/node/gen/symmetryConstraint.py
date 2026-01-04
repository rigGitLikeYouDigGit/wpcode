

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
	node : SymmetryConstraint = None
	pass
class ConstraintJointOrientYPlug(Plug):
	parent : ConstraintJointOrientPlug = PlugDescriptor("constraintJointOrient")
	node : SymmetryConstraint = None
	pass
class ConstraintJointOrientZPlug(Plug):
	parent : ConstraintJointOrientPlug = PlugDescriptor("constraintJointOrient")
	node : SymmetryConstraint = None
	pass
class ConstraintJointOrientPlug(Plug):
	parent : ConstrainedPlug = PlugDescriptor("constrained")
	constraintJointOrientX_ : ConstraintJointOrientXPlug = PlugDescriptor("constraintJointOrientX")
	cjx_ : ConstraintJointOrientXPlug = PlugDescriptor("constraintJointOrientX")
	constraintJointOrientY_ : ConstraintJointOrientYPlug = PlugDescriptor("constraintJointOrientY")
	cjy_ : ConstraintJointOrientYPlug = PlugDescriptor("constraintJointOrientY")
	constraintJointOrientZ_ : ConstraintJointOrientZPlug = PlugDescriptor("constraintJointOrientZ")
	cjz_ : ConstraintJointOrientZPlug = PlugDescriptor("constraintJointOrientZ")
	node : SymmetryConstraint = None
	pass
class ConstraintRotateXPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : SymmetryConstraint = None
	pass
class ConstraintRotateYPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : SymmetryConstraint = None
	pass
class ConstraintRotateZPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : SymmetryConstraint = None
	pass
class ConstraintRotatePlug(Plug):
	parent : ConstrainedPlug = PlugDescriptor("constrained")
	constraintRotateX_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	crx_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	constraintRotateY_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	cry_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	constraintRotateZ_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	crz_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	node : SymmetryConstraint = None
	pass
class ConstraintRotateOrderPlug(Plug):
	parent : ConstrainedPlug = PlugDescriptor("constrained")
	node : SymmetryConstraint = None
	pass
class ConstraintScaleXPlug(Plug):
	parent : ConstraintScalePlug = PlugDescriptor("constraintScale")
	node : SymmetryConstraint = None
	pass
class ConstraintScaleYPlug(Plug):
	parent : ConstraintScalePlug = PlugDescriptor("constraintScale")
	node : SymmetryConstraint = None
	pass
class ConstraintScaleZPlug(Plug):
	parent : ConstraintScalePlug = PlugDescriptor("constraintScale")
	node : SymmetryConstraint = None
	pass
class ConstraintScalePlug(Plug):
	parent : ConstrainedPlug = PlugDescriptor("constrained")
	constraintScaleX_ : ConstraintScaleXPlug = PlugDescriptor("constraintScaleX")
	csx_ : ConstraintScaleXPlug = PlugDescriptor("constraintScaleX")
	constraintScaleY_ : ConstraintScaleYPlug = PlugDescriptor("constraintScaleY")
	csy_ : ConstraintScaleYPlug = PlugDescriptor("constraintScaleY")
	constraintScaleZ_ : ConstraintScaleZPlug = PlugDescriptor("constraintScaleZ")
	csz_ : ConstraintScaleZPlug = PlugDescriptor("constraintScaleZ")
	node : SymmetryConstraint = None
	pass
class ConstraintTranslateXPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : SymmetryConstraint = None
	pass
class ConstraintTranslateYPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : SymmetryConstraint = None
	pass
class ConstraintTranslateZPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : SymmetryConstraint = None
	pass
class ConstraintTranslatePlug(Plug):
	parent : ConstrainedPlug = PlugDescriptor("constrained")
	constraintTranslateX_ : ConstraintTranslateXPlug = PlugDescriptor("constraintTranslateX")
	ctx_ : ConstraintTranslateXPlug = PlugDescriptor("constraintTranslateX")
	constraintTranslateY_ : ConstraintTranslateYPlug = PlugDescriptor("constraintTranslateY")
	cty_ : ConstraintTranslateYPlug = PlugDescriptor("constraintTranslateY")
	constraintTranslateZ_ : ConstraintTranslateZPlug = PlugDescriptor("constraintTranslateZ")
	ctz_ : ConstraintTranslateZPlug = PlugDescriptor("constraintTranslateZ")
	node : SymmetryConstraint = None
	pass
class ConstrainedPlug(Plug):
	constraintJointOrient_ : ConstraintJointOrientPlug = PlugDescriptor("constraintJointOrient")
	cjo_ : ConstraintJointOrientPlug = PlugDescriptor("constraintJointOrient")
	constraintRotate_ : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	cr_ : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	constraintRotateOrder_ : ConstraintRotateOrderPlug = PlugDescriptor("constraintRotateOrder")
	cro_ : ConstraintRotateOrderPlug = PlugDescriptor("constraintRotateOrder")
	constraintScale_ : ConstraintScalePlug = PlugDescriptor("constraintScale")
	cs_ : ConstraintScalePlug = PlugDescriptor("constraintScale")
	constraintTranslate_ : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	ct_ : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : SymmetryConstraint = None
	pass
class ConstraintInverseParentWorldMatrixPlug(Plug):
	node : SymmetryConstraint = None
	pass
class SymmetryMiddlePointXPlug(Plug):
	parent : SymmetryMiddlePointPlug = PlugDescriptor("symmetryMiddlePoint")
	node : SymmetryConstraint = None
	pass
class SymmetryMiddlePointYPlug(Plug):
	parent : SymmetryMiddlePointPlug = PlugDescriptor("symmetryMiddlePoint")
	node : SymmetryConstraint = None
	pass
class SymmetryMiddlePointZPlug(Plug):
	parent : SymmetryMiddlePointPlug = PlugDescriptor("symmetryMiddlePoint")
	node : SymmetryConstraint = None
	pass
class SymmetryMiddlePointPlug(Plug):
	symmetryMiddlePointX_ : SymmetryMiddlePointXPlug = PlugDescriptor("symmetryMiddlePointX")
	cmpx_ : SymmetryMiddlePointXPlug = PlugDescriptor("symmetryMiddlePointX")
	symmetryMiddlePointY_ : SymmetryMiddlePointYPlug = PlugDescriptor("symmetryMiddlePointY")
	cmpy_ : SymmetryMiddlePointYPlug = PlugDescriptor("symmetryMiddlePointY")
	symmetryMiddlePointZ_ : SymmetryMiddlePointZPlug = PlugDescriptor("symmetryMiddlePointZ")
	cmpz_ : SymmetryMiddlePointZPlug = PlugDescriptor("symmetryMiddlePointZ")
	node : SymmetryConstraint = None
	pass
class SymmetryRootOffsetXPlug(Plug):
	parent : SymmetryRootOffsetPlug = PlugDescriptor("symmetryRootOffset")
	node : SymmetryConstraint = None
	pass
class SymmetryRootOffsetYPlug(Plug):
	parent : SymmetryRootOffsetPlug = PlugDescriptor("symmetryRootOffset")
	node : SymmetryConstraint = None
	pass
class SymmetryRootOffsetZPlug(Plug):
	parent : SymmetryRootOffsetPlug = PlugDescriptor("symmetryRootOffset")
	node : SymmetryConstraint = None
	pass
class SymmetryRootOffsetPlug(Plug):
	symmetryRootOffsetX_ : SymmetryRootOffsetXPlug = PlugDescriptor("symmetryRootOffsetX")
	srox_ : SymmetryRootOffsetXPlug = PlugDescriptor("symmetryRootOffsetX")
	symmetryRootOffsetY_ : SymmetryRootOffsetYPlug = PlugDescriptor("symmetryRootOffsetY")
	sroy_ : SymmetryRootOffsetYPlug = PlugDescriptor("symmetryRootOffsetY")
	symmetryRootOffsetZ_ : SymmetryRootOffsetZPlug = PlugDescriptor("symmetryRootOffsetZ")
	sroz_ : SymmetryRootOffsetZPlug = PlugDescriptor("symmetryRootOffsetZ")
	node : SymmetryConstraint = None
	pass
class SymmetryRootWorldMatrixPlug(Plug):
	node : SymmetryConstraint = None
	pass
class TargetChildTranslateXPlug(Plug):
	parent : TargetChildTranslatePlug = PlugDescriptor("targetChildTranslate")
	node : SymmetryConstraint = None
	pass
class TargetChildTranslateYPlug(Plug):
	parent : TargetChildTranslatePlug = PlugDescriptor("targetChildTranslate")
	node : SymmetryConstraint = None
	pass
class TargetChildTranslateZPlug(Plug):
	parent : TargetChildTranslatePlug = PlugDescriptor("targetChildTranslate")
	node : SymmetryConstraint = None
	pass
class TargetChildTranslatePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetChildTranslateX_ : TargetChildTranslateXPlug = PlugDescriptor("targetChildTranslateX")
	tcx_ : TargetChildTranslateXPlug = PlugDescriptor("targetChildTranslateX")
	targetChildTranslateY_ : TargetChildTranslateYPlug = PlugDescriptor("targetChildTranslateY")
	tcy_ : TargetChildTranslateYPlug = PlugDescriptor("targetChildTranslateY")
	targetChildTranslateZ_ : TargetChildTranslateZPlug = PlugDescriptor("targetChildTranslateZ")
	tcz_ : TargetChildTranslateZPlug = PlugDescriptor("targetChildTranslateZ")
	node : SymmetryConstraint = None
	pass
class TargetJointOrientXPlug(Plug):
	parent : TargetJointOrientPlug = PlugDescriptor("targetJointOrient")
	node : SymmetryConstraint = None
	pass
class TargetJointOrientYPlug(Plug):
	parent : TargetJointOrientPlug = PlugDescriptor("targetJointOrient")
	node : SymmetryConstraint = None
	pass
class TargetJointOrientZPlug(Plug):
	parent : TargetJointOrientPlug = PlugDescriptor("targetJointOrient")
	node : SymmetryConstraint = None
	pass
class TargetJointOrientPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetJointOrientX_ : TargetJointOrientXPlug = PlugDescriptor("targetJointOrientX")
	tjx_ : TargetJointOrientXPlug = PlugDescriptor("targetJointOrientX")
	targetJointOrientY_ : TargetJointOrientYPlug = PlugDescriptor("targetJointOrientY")
	tjy_ : TargetJointOrientYPlug = PlugDescriptor("targetJointOrientY")
	targetJointOrientZ_ : TargetJointOrientZPlug = PlugDescriptor("targetJointOrientZ")
	tjz_ : TargetJointOrientZPlug = PlugDescriptor("targetJointOrientZ")
	node : SymmetryConstraint = None
	pass
class TargetJointOrientTypePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : SymmetryConstraint = None
	pass
class TargetParentMatrixPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : SymmetryConstraint = None
	pass
class TargetRotateXPlug(Plug):
	parent : TargetRotatePlug = PlugDescriptor("targetRotate")
	node : SymmetryConstraint = None
	pass
class TargetRotateYPlug(Plug):
	parent : TargetRotatePlug = PlugDescriptor("targetRotate")
	node : SymmetryConstraint = None
	pass
class TargetRotateZPlug(Plug):
	parent : TargetRotatePlug = PlugDescriptor("targetRotate")
	node : SymmetryConstraint = None
	pass
class TargetRotatePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetRotateX_ : TargetRotateXPlug = PlugDescriptor("targetRotateX")
	trx_ : TargetRotateXPlug = PlugDescriptor("targetRotateX")
	targetRotateY_ : TargetRotateYPlug = PlugDescriptor("targetRotateY")
	try_ : TargetRotateYPlug = PlugDescriptor("targetRotateY")
	targetRotateZ_ : TargetRotateZPlug = PlugDescriptor("targetRotateZ")
	trz_ : TargetRotateZPlug = PlugDescriptor("targetRotateZ")
	node : SymmetryConstraint = None
	pass
class TargetRotateOrderPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : SymmetryConstraint = None
	pass
class TargetScaleXPlug(Plug):
	parent : TargetScalePlug = PlugDescriptor("targetScale")
	node : SymmetryConstraint = None
	pass
class TargetScaleYPlug(Plug):
	parent : TargetScalePlug = PlugDescriptor("targetScale")
	node : SymmetryConstraint = None
	pass
class TargetScaleZPlug(Plug):
	parent : TargetScalePlug = PlugDescriptor("targetScale")
	node : SymmetryConstraint = None
	pass
class TargetScalePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetScaleX_ : TargetScaleXPlug = PlugDescriptor("targetScaleX")
	tsx_ : TargetScaleXPlug = PlugDescriptor("targetScaleX")
	targetScaleY_ : TargetScaleYPlug = PlugDescriptor("targetScaleY")
	tsy_ : TargetScaleYPlug = PlugDescriptor("targetScaleY")
	targetScaleZ_ : TargetScaleZPlug = PlugDescriptor("targetScaleZ")
	tsz_ : TargetScaleZPlug = PlugDescriptor("targetScaleZ")
	node : SymmetryConstraint = None
	pass
class TargetTranslateXPlug(Plug):
	parent : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	node : SymmetryConstraint = None
	pass
class TargetTranslateYPlug(Plug):
	parent : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	node : SymmetryConstraint = None
	pass
class TargetTranslateZPlug(Plug):
	parent : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	node : SymmetryConstraint = None
	pass
class TargetTranslatePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetTranslateX_ : TargetTranslateXPlug = PlugDescriptor("targetTranslateX")
	ttx_ : TargetTranslateXPlug = PlugDescriptor("targetTranslateX")
	targetTranslateY_ : TargetTranslateYPlug = PlugDescriptor("targetTranslateY")
	tty_ : TargetTranslateYPlug = PlugDescriptor("targetTranslateY")
	targetTranslateZ_ : TargetTranslateZPlug = PlugDescriptor("targetTranslateZ")
	ttz_ : TargetTranslateZPlug = PlugDescriptor("targetTranslateZ")
	node : SymmetryConstraint = None
	pass
class TargetWorldMatrixPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : SymmetryConstraint = None
	pass
class TargetPlug(Plug):
	targetChildTranslate_ : TargetChildTranslatePlug = PlugDescriptor("targetChildTranslate")
	tct_ : TargetChildTranslatePlug = PlugDescriptor("targetChildTranslate")
	targetJointOrient_ : TargetJointOrientPlug = PlugDescriptor("targetJointOrient")
	tjo_ : TargetJointOrientPlug = PlugDescriptor("targetJointOrient")
	targetJointOrientType_ : TargetJointOrientTypePlug = PlugDescriptor("targetJointOrientType")
	tjt_ : TargetJointOrientTypePlug = PlugDescriptor("targetJointOrientType")
	targetParentMatrix_ : TargetParentMatrixPlug = PlugDescriptor("targetParentMatrix")
	tpm_ : TargetParentMatrixPlug = PlugDescriptor("targetParentMatrix")
	targetRotate_ : TargetRotatePlug = PlugDescriptor("targetRotate")
	tr_ : TargetRotatePlug = PlugDescriptor("targetRotate")
	targetRotateOrder_ : TargetRotateOrderPlug = PlugDescriptor("targetRotateOrder")
	tro_ : TargetRotateOrderPlug = PlugDescriptor("targetRotateOrder")
	targetScale_ : TargetScalePlug = PlugDescriptor("targetScale")
	ts_ : TargetScalePlug = PlugDescriptor("targetScale")
	targetTranslate_ : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	tt_ : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	targetWorldMatrix_ : TargetWorldMatrixPlug = PlugDescriptor("targetWorldMatrix")
	twm_ : TargetWorldMatrixPlug = PlugDescriptor("targetWorldMatrix")
	node : SymmetryConstraint = None
	pass
class XAxisPlug(Plug):
	node : SymmetryConstraint = None
	pass
class XChildAxisPlug(Plug):
	node : SymmetryConstraint = None
	pass
class YAxisPlug(Plug):
	node : SymmetryConstraint = None
	pass
class YChildAxisPlug(Plug):
	node : SymmetryConstraint = None
	pass
class ZAxisPlug(Plug):
	node : SymmetryConstraint = None
	pass
class ZChildAxisPlug(Plug):
	node : SymmetryConstraint = None
	pass
# endregion


# define node class
class SymmetryConstraint(Constraint):
	constraintJointOrientX_ : ConstraintJointOrientXPlug = PlugDescriptor("constraintJointOrientX")
	constraintJointOrientY_ : ConstraintJointOrientYPlug = PlugDescriptor("constraintJointOrientY")
	constraintJointOrientZ_ : ConstraintJointOrientZPlug = PlugDescriptor("constraintJointOrientZ")
	constraintJointOrient_ : ConstraintJointOrientPlug = PlugDescriptor("constraintJointOrient")
	constraintRotateX_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	constraintRotateY_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	constraintRotateZ_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	constraintRotate_ : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	constraintRotateOrder_ : ConstraintRotateOrderPlug = PlugDescriptor("constraintRotateOrder")
	constraintScaleX_ : ConstraintScaleXPlug = PlugDescriptor("constraintScaleX")
	constraintScaleY_ : ConstraintScaleYPlug = PlugDescriptor("constraintScaleY")
	constraintScaleZ_ : ConstraintScaleZPlug = PlugDescriptor("constraintScaleZ")
	constraintScale_ : ConstraintScalePlug = PlugDescriptor("constraintScale")
	constraintTranslateX_ : ConstraintTranslateXPlug = PlugDescriptor("constraintTranslateX")
	constraintTranslateY_ : ConstraintTranslateYPlug = PlugDescriptor("constraintTranslateY")
	constraintTranslateZ_ : ConstraintTranslateZPlug = PlugDescriptor("constraintTranslateZ")
	constraintTranslate_ : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	constrained_ : ConstrainedPlug = PlugDescriptor("constrained")
	constraintInverseParentWorldMatrix_ : ConstraintInverseParentWorldMatrixPlug = PlugDescriptor("constraintInverseParentWorldMatrix")
	symmetryMiddlePointX_ : SymmetryMiddlePointXPlug = PlugDescriptor("symmetryMiddlePointX")
	symmetryMiddlePointY_ : SymmetryMiddlePointYPlug = PlugDescriptor("symmetryMiddlePointY")
	symmetryMiddlePointZ_ : SymmetryMiddlePointZPlug = PlugDescriptor("symmetryMiddlePointZ")
	symmetryMiddlePoint_ : SymmetryMiddlePointPlug = PlugDescriptor("symmetryMiddlePoint")
	symmetryRootOffsetX_ : SymmetryRootOffsetXPlug = PlugDescriptor("symmetryRootOffsetX")
	symmetryRootOffsetY_ : SymmetryRootOffsetYPlug = PlugDescriptor("symmetryRootOffsetY")
	symmetryRootOffsetZ_ : SymmetryRootOffsetZPlug = PlugDescriptor("symmetryRootOffsetZ")
	symmetryRootOffset_ : SymmetryRootOffsetPlug = PlugDescriptor("symmetryRootOffset")
	symmetryRootWorldMatrix_ : SymmetryRootWorldMatrixPlug = PlugDescriptor("symmetryRootWorldMatrix")
	targetChildTranslateX_ : TargetChildTranslateXPlug = PlugDescriptor("targetChildTranslateX")
	targetChildTranslateY_ : TargetChildTranslateYPlug = PlugDescriptor("targetChildTranslateY")
	targetChildTranslateZ_ : TargetChildTranslateZPlug = PlugDescriptor("targetChildTranslateZ")
	targetChildTranslate_ : TargetChildTranslatePlug = PlugDescriptor("targetChildTranslate")
	targetJointOrientX_ : TargetJointOrientXPlug = PlugDescriptor("targetJointOrientX")
	targetJointOrientY_ : TargetJointOrientYPlug = PlugDescriptor("targetJointOrientY")
	targetJointOrientZ_ : TargetJointOrientZPlug = PlugDescriptor("targetJointOrientZ")
	targetJointOrient_ : TargetJointOrientPlug = PlugDescriptor("targetJointOrient")
	targetJointOrientType_ : TargetJointOrientTypePlug = PlugDescriptor("targetJointOrientType")
	targetParentMatrix_ : TargetParentMatrixPlug = PlugDescriptor("targetParentMatrix")
	targetRotateX_ : TargetRotateXPlug = PlugDescriptor("targetRotateX")
	targetRotateY_ : TargetRotateYPlug = PlugDescriptor("targetRotateY")
	targetRotateZ_ : TargetRotateZPlug = PlugDescriptor("targetRotateZ")
	targetRotate_ : TargetRotatePlug = PlugDescriptor("targetRotate")
	targetRotateOrder_ : TargetRotateOrderPlug = PlugDescriptor("targetRotateOrder")
	targetScaleX_ : TargetScaleXPlug = PlugDescriptor("targetScaleX")
	targetScaleY_ : TargetScaleYPlug = PlugDescriptor("targetScaleY")
	targetScaleZ_ : TargetScaleZPlug = PlugDescriptor("targetScaleZ")
	targetScale_ : TargetScalePlug = PlugDescriptor("targetScale")
	targetTranslateX_ : TargetTranslateXPlug = PlugDescriptor("targetTranslateX")
	targetTranslateY_ : TargetTranslateYPlug = PlugDescriptor("targetTranslateY")
	targetTranslateZ_ : TargetTranslateZPlug = PlugDescriptor("targetTranslateZ")
	targetTranslate_ : TargetTranslatePlug = PlugDescriptor("targetTranslate")
	targetWorldMatrix_ : TargetWorldMatrixPlug = PlugDescriptor("targetWorldMatrix")
	target_ : TargetPlug = PlugDescriptor("target")
	xAxis_ : XAxisPlug = PlugDescriptor("xAxis")
	xChildAxis_ : XChildAxisPlug = PlugDescriptor("xChildAxis")
	yAxis_ : YAxisPlug = PlugDescriptor("yAxis")
	yChildAxis_ : YChildAxisPlug = PlugDescriptor("yChildAxis")
	zAxis_ : ZAxisPlug = PlugDescriptor("zAxis")
	zChildAxis_ : ZChildAxisPlug = PlugDescriptor("zChildAxis")

	# node attributes

	typeName = "symmetryConstraint"
	apiTypeInt = 241
	apiTypeStr = "kSymmetryConstraint"
	typeIdInt = 1146309955
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["constraintJointOrientX", "constraintJointOrientY", "constraintJointOrientZ", "constraintJointOrient", "constraintRotateX", "constraintRotateY", "constraintRotateZ", "constraintRotate", "constraintRotateOrder", "constraintScaleX", "constraintScaleY", "constraintScaleZ", "constraintScale", "constraintTranslateX", "constraintTranslateY", "constraintTranslateZ", "constraintTranslate", "constrained", "constraintInverseParentWorldMatrix", "symmetryMiddlePointX", "symmetryMiddlePointY", "symmetryMiddlePointZ", "symmetryMiddlePoint", "symmetryRootOffsetX", "symmetryRootOffsetY", "symmetryRootOffsetZ", "symmetryRootOffset", "symmetryRootWorldMatrix", "targetChildTranslateX", "targetChildTranslateY", "targetChildTranslateZ", "targetChildTranslate", "targetJointOrientX", "targetJointOrientY", "targetJointOrientZ", "targetJointOrient", "targetJointOrientType", "targetParentMatrix", "targetRotateX", "targetRotateY", "targetRotateZ", "targetRotate", "targetRotateOrder", "targetScaleX", "targetScaleY", "targetScaleZ", "targetScale", "targetTranslateX", "targetTranslateY", "targetTranslateZ", "targetTranslate", "targetWorldMatrix", "target", "xAxis", "xChildAxis", "yAxis", "yChildAxis", "zAxis", "zChildAxis"]
	nodeLeafPlugs = ["constrained", "constraintInverseParentWorldMatrix", "symmetryMiddlePoint", "symmetryRootOffset", "symmetryRootWorldMatrix", "target", "xAxis", "xChildAxis", "yAxis", "yChildAxis", "zAxis", "zChildAxis"]
	pass

