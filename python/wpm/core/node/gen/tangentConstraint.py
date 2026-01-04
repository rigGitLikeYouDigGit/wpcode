

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
class AimVectorXPlug(Plug):
	parent : AimVectorPlug = PlugDescriptor("aimVector")
	node : TangentConstraint = None
	pass
class AimVectorYPlug(Plug):
	parent : AimVectorPlug = PlugDescriptor("aimVector")
	node : TangentConstraint = None
	pass
class AimVectorZPlug(Plug):
	parent : AimVectorPlug = PlugDescriptor("aimVector")
	node : TangentConstraint = None
	pass
class AimVectorPlug(Plug):
	aimVectorX_ : AimVectorXPlug = PlugDescriptor("aimVectorX")
	ax_ : AimVectorXPlug = PlugDescriptor("aimVectorX")
	aimVectorY_ : AimVectorYPlug = PlugDescriptor("aimVectorY")
	ay_ : AimVectorYPlug = PlugDescriptor("aimVectorY")
	aimVectorZ_ : AimVectorZPlug = PlugDescriptor("aimVectorZ")
	az_ : AimVectorZPlug = PlugDescriptor("aimVectorZ")
	node : TangentConstraint = None
	pass
class ConstraintJointOrientXPlug(Plug):
	parent : ConstraintJointOrientPlug = PlugDescriptor("constraintJointOrient")
	node : TangentConstraint = None
	pass
class ConstraintJointOrientYPlug(Plug):
	parent : ConstraintJointOrientPlug = PlugDescriptor("constraintJointOrient")
	node : TangentConstraint = None
	pass
class ConstraintJointOrientZPlug(Plug):
	parent : ConstraintJointOrientPlug = PlugDescriptor("constraintJointOrient")
	node : TangentConstraint = None
	pass
class ConstraintJointOrientPlug(Plug):
	constraintJointOrientX_ : ConstraintJointOrientXPlug = PlugDescriptor("constraintJointOrientX")
	cjox_ : ConstraintJointOrientXPlug = PlugDescriptor("constraintJointOrientX")
	constraintJointOrientY_ : ConstraintJointOrientYPlug = PlugDescriptor("constraintJointOrientY")
	cjoy_ : ConstraintJointOrientYPlug = PlugDescriptor("constraintJointOrientY")
	constraintJointOrientZ_ : ConstraintJointOrientZPlug = PlugDescriptor("constraintJointOrientZ")
	cjoz_ : ConstraintJointOrientZPlug = PlugDescriptor("constraintJointOrientZ")
	node : TangentConstraint = None
	pass
class ConstraintParentInverseMatrixPlug(Plug):
	node : TangentConstraint = None
	pass
class ConstraintRotateXPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : TangentConstraint = None
	pass
class ConstraintRotateYPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : TangentConstraint = None
	pass
class ConstraintRotateZPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : TangentConstraint = None
	pass
class ConstraintRotatePlug(Plug):
	constraintRotateX_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	crx_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	constraintRotateY_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	cry_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	constraintRotateZ_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	crz_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	node : TangentConstraint = None
	pass
class ConstraintRotateOrderPlug(Plug):
	node : TangentConstraint = None
	pass
class ConstraintRotatePivotXPlug(Plug):
	parent : ConstraintRotatePivotPlug = PlugDescriptor("constraintRotatePivot")
	node : TangentConstraint = None
	pass
class ConstraintRotatePivotYPlug(Plug):
	parent : ConstraintRotatePivotPlug = PlugDescriptor("constraintRotatePivot")
	node : TangentConstraint = None
	pass
class ConstraintRotatePivotZPlug(Plug):
	parent : ConstraintRotatePivotPlug = PlugDescriptor("constraintRotatePivot")
	node : TangentConstraint = None
	pass
class ConstraintRotatePivotPlug(Plug):
	constraintRotatePivotX_ : ConstraintRotatePivotXPlug = PlugDescriptor("constraintRotatePivotX")
	crpx_ : ConstraintRotatePivotXPlug = PlugDescriptor("constraintRotatePivotX")
	constraintRotatePivotY_ : ConstraintRotatePivotYPlug = PlugDescriptor("constraintRotatePivotY")
	crpy_ : ConstraintRotatePivotYPlug = PlugDescriptor("constraintRotatePivotY")
	constraintRotatePivotZ_ : ConstraintRotatePivotZPlug = PlugDescriptor("constraintRotatePivotZ")
	crpz_ : ConstraintRotatePivotZPlug = PlugDescriptor("constraintRotatePivotZ")
	node : TangentConstraint = None
	pass
class ConstraintRotateTranslateXPlug(Plug):
	parent : ConstraintRotateTranslatePlug = PlugDescriptor("constraintRotateTranslate")
	node : TangentConstraint = None
	pass
class ConstraintRotateTranslateYPlug(Plug):
	parent : ConstraintRotateTranslatePlug = PlugDescriptor("constraintRotateTranslate")
	node : TangentConstraint = None
	pass
class ConstraintRotateTranslateZPlug(Plug):
	parent : ConstraintRotateTranslatePlug = PlugDescriptor("constraintRotateTranslate")
	node : TangentConstraint = None
	pass
class ConstraintRotateTranslatePlug(Plug):
	constraintRotateTranslateX_ : ConstraintRotateTranslateXPlug = PlugDescriptor("constraintRotateTranslateX")
	crtx_ : ConstraintRotateTranslateXPlug = PlugDescriptor("constraintRotateTranslateX")
	constraintRotateTranslateY_ : ConstraintRotateTranslateYPlug = PlugDescriptor("constraintRotateTranslateY")
	crty_ : ConstraintRotateTranslateYPlug = PlugDescriptor("constraintRotateTranslateY")
	constraintRotateTranslateZ_ : ConstraintRotateTranslateZPlug = PlugDescriptor("constraintRotateTranslateZ")
	crtz_ : ConstraintRotateTranslateZPlug = PlugDescriptor("constraintRotateTranslateZ")
	node : TangentConstraint = None
	pass
class ConstraintTranslateXPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : TangentConstraint = None
	pass
class ConstraintTranslateYPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : TangentConstraint = None
	pass
class ConstraintTranslateZPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : TangentConstraint = None
	pass
class ConstraintTranslatePlug(Plug):
	constraintTranslateX_ : ConstraintTranslateXPlug = PlugDescriptor("constraintTranslateX")
	ctx_ : ConstraintTranslateXPlug = PlugDescriptor("constraintTranslateX")
	constraintTranslateY_ : ConstraintTranslateYPlug = PlugDescriptor("constraintTranslateY")
	cty_ : ConstraintTranslateYPlug = PlugDescriptor("constraintTranslateY")
	constraintTranslateZ_ : ConstraintTranslateZPlug = PlugDescriptor("constraintTranslateZ")
	ctz_ : ConstraintTranslateZPlug = PlugDescriptor("constraintTranslateZ")
	node : TangentConstraint = None
	pass
class ConstraintVectorXPlug(Plug):
	parent : ConstraintVectorPlug = PlugDescriptor("constraintVector")
	node : TangentConstraint = None
	pass
class ConstraintVectorYPlug(Plug):
	parent : ConstraintVectorPlug = PlugDescriptor("constraintVector")
	node : TangentConstraint = None
	pass
class ConstraintVectorZPlug(Plug):
	parent : ConstraintVectorPlug = PlugDescriptor("constraintVector")
	node : TangentConstraint = None
	pass
class ConstraintVectorPlug(Plug):
	constraintVectorX_ : ConstraintVectorXPlug = PlugDescriptor("constraintVectorX")
	cvx_ : ConstraintVectorXPlug = PlugDescriptor("constraintVectorX")
	constraintVectorY_ : ConstraintVectorYPlug = PlugDescriptor("constraintVectorY")
	cvy_ : ConstraintVectorYPlug = PlugDescriptor("constraintVectorY")
	constraintVectorZ_ : ConstraintVectorZPlug = PlugDescriptor("constraintVectorZ")
	cvz_ : ConstraintVectorZPlug = PlugDescriptor("constraintVectorZ")
	node : TangentConstraint = None
	pass
class RestRotateXPlug(Plug):
	parent : RestRotatePlug = PlugDescriptor("restRotate")
	node : TangentConstraint = None
	pass
class RestRotateYPlug(Plug):
	parent : RestRotatePlug = PlugDescriptor("restRotate")
	node : TangentConstraint = None
	pass
class RestRotateZPlug(Plug):
	parent : RestRotatePlug = PlugDescriptor("restRotate")
	node : TangentConstraint = None
	pass
class RestRotatePlug(Plug):
	restRotateX_ : RestRotateXPlug = PlugDescriptor("restRotateX")
	rrx_ : RestRotateXPlug = PlugDescriptor("restRotateX")
	restRotateY_ : RestRotateYPlug = PlugDescriptor("restRotateY")
	rry_ : RestRotateYPlug = PlugDescriptor("restRotateY")
	restRotateZ_ : RestRotateZPlug = PlugDescriptor("restRotateZ")
	rrz_ : RestRotateZPlug = PlugDescriptor("restRotateZ")
	node : TangentConstraint = None
	pass
class TargetGeometryPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : TangentConstraint = None
	pass
class TargetWeightPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : TangentConstraint = None
	pass
class TargetPlug(Plug):
	targetGeometry_ : TargetGeometryPlug = PlugDescriptor("targetGeometry")
	tgm_ : TargetGeometryPlug = PlugDescriptor("targetGeometry")
	targetWeight_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	tw_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	node : TangentConstraint = None
	pass
class UpVectorXPlug(Plug):
	parent : UpVectorPlug = PlugDescriptor("upVector")
	node : TangentConstraint = None
	pass
class UpVectorYPlug(Plug):
	parent : UpVectorPlug = PlugDescriptor("upVector")
	node : TangentConstraint = None
	pass
class UpVectorZPlug(Plug):
	parent : UpVectorPlug = PlugDescriptor("upVector")
	node : TangentConstraint = None
	pass
class UpVectorPlug(Plug):
	upVectorX_ : UpVectorXPlug = PlugDescriptor("upVectorX")
	ux_ : UpVectorXPlug = PlugDescriptor("upVectorX")
	upVectorY_ : UpVectorYPlug = PlugDescriptor("upVectorY")
	uy_ : UpVectorYPlug = PlugDescriptor("upVectorY")
	upVectorZ_ : UpVectorZPlug = PlugDescriptor("upVectorZ")
	uz_ : UpVectorZPlug = PlugDescriptor("upVectorZ")
	node : TangentConstraint = None
	pass
class WorldUpMatrixPlug(Plug):
	node : TangentConstraint = None
	pass
class WorldUpTypePlug(Plug):
	node : TangentConstraint = None
	pass
class WorldUpVectorXPlug(Plug):
	parent : WorldUpVectorPlug = PlugDescriptor("worldUpVector")
	node : TangentConstraint = None
	pass
class WorldUpVectorYPlug(Plug):
	parent : WorldUpVectorPlug = PlugDescriptor("worldUpVector")
	node : TangentConstraint = None
	pass
class WorldUpVectorZPlug(Plug):
	parent : WorldUpVectorPlug = PlugDescriptor("worldUpVector")
	node : TangentConstraint = None
	pass
class WorldUpVectorPlug(Plug):
	worldUpVectorX_ : WorldUpVectorXPlug = PlugDescriptor("worldUpVectorX")
	wux_ : WorldUpVectorXPlug = PlugDescriptor("worldUpVectorX")
	worldUpVectorY_ : WorldUpVectorYPlug = PlugDescriptor("worldUpVectorY")
	wuy_ : WorldUpVectorYPlug = PlugDescriptor("worldUpVectorY")
	worldUpVectorZ_ : WorldUpVectorZPlug = PlugDescriptor("worldUpVectorZ")
	wuz_ : WorldUpVectorZPlug = PlugDescriptor("worldUpVectorZ")
	node : TangentConstraint = None
	pass
# endregion


# define node class
class TangentConstraint(Constraint):
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
	restRotateX_ : RestRotateXPlug = PlugDescriptor("restRotateX")
	restRotateY_ : RestRotateYPlug = PlugDescriptor("restRotateY")
	restRotateZ_ : RestRotateZPlug = PlugDescriptor("restRotateZ")
	restRotate_ : RestRotatePlug = PlugDescriptor("restRotate")
	targetGeometry_ : TargetGeometryPlug = PlugDescriptor("targetGeometry")
	targetWeight_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	target_ : TargetPlug = PlugDescriptor("target")
	upVectorX_ : UpVectorXPlug = PlugDescriptor("upVectorX")
	upVectorY_ : UpVectorYPlug = PlugDescriptor("upVectorY")
	upVectorZ_ : UpVectorZPlug = PlugDescriptor("upVectorZ")
	upVector_ : UpVectorPlug = PlugDescriptor("upVector")
	worldUpMatrix_ : WorldUpMatrixPlug = PlugDescriptor("worldUpMatrix")
	worldUpType_ : WorldUpTypePlug = PlugDescriptor("worldUpType")
	worldUpVectorX_ : WorldUpVectorXPlug = PlugDescriptor("worldUpVectorX")
	worldUpVectorY_ : WorldUpVectorYPlug = PlugDescriptor("worldUpVectorY")
	worldUpVectorZ_ : WorldUpVectorZPlug = PlugDescriptor("worldUpVectorZ")
	worldUpVector_ : WorldUpVectorPlug = PlugDescriptor("worldUpVector")

	# node attributes

	typeName = "tangentConstraint"
	apiTypeInt = 245
	apiTypeStr = "kTangentConstraint"
	typeIdInt = 1146372914
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["aimVectorX", "aimVectorY", "aimVectorZ", "aimVector", "constraintJointOrientX", "constraintJointOrientY", "constraintJointOrientZ", "constraintJointOrient", "constraintParentInverseMatrix", "constraintRotateX", "constraintRotateY", "constraintRotateZ", "constraintRotate", "constraintRotateOrder", "constraintRotatePivotX", "constraintRotatePivotY", "constraintRotatePivotZ", "constraintRotatePivot", "constraintRotateTranslateX", "constraintRotateTranslateY", "constraintRotateTranslateZ", "constraintRotateTranslate", "constraintTranslateX", "constraintTranslateY", "constraintTranslateZ", "constraintTranslate", "constraintVectorX", "constraintVectorY", "constraintVectorZ", "constraintVector", "restRotateX", "restRotateY", "restRotateZ", "restRotate", "targetGeometry", "targetWeight", "target", "upVectorX", "upVectorY", "upVectorZ", "upVector", "worldUpMatrix", "worldUpType", "worldUpVectorX", "worldUpVectorY", "worldUpVectorZ", "worldUpVector"]
	nodeLeafPlugs = ["aimVector", "constraintJointOrient", "constraintParentInverseMatrix", "constraintRotate", "constraintRotateOrder", "constraintRotatePivot", "constraintRotateTranslate", "constraintTranslate", "constraintVector", "restRotate", "target", "upVector", "worldUpMatrix", "worldUpType", "worldUpVector"]
	pass

