

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
class ConstraintParentInverseMatrixPlug(Plug):
	node : PointOnPolyConstraint = None
	pass
class ConstraintRotateXPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : PointOnPolyConstraint = None
	pass
class ConstraintRotateYPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : PointOnPolyConstraint = None
	pass
class ConstraintRotateZPlug(Plug):
	parent : ConstraintRotatePlug = PlugDescriptor("constraintRotate")
	node : PointOnPolyConstraint = None
	pass
class ConstraintRotatePlug(Plug):
	constraintRotateX_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	crx_ : ConstraintRotateXPlug = PlugDescriptor("constraintRotateX")
	constraintRotateY_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	cry_ : ConstraintRotateYPlug = PlugDescriptor("constraintRotateY")
	constraintRotateZ_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	crz_ : ConstraintRotateZPlug = PlugDescriptor("constraintRotateZ")
	node : PointOnPolyConstraint = None
	pass
class ConstraintRotateOrderPlug(Plug):
	node : PointOnPolyConstraint = None
	pass
class ConstraintRotatePivotXPlug(Plug):
	parent : ConstraintRotatePivotPlug = PlugDescriptor("constraintRotatePivot")
	node : PointOnPolyConstraint = None
	pass
class ConstraintRotatePivotYPlug(Plug):
	parent : ConstraintRotatePivotPlug = PlugDescriptor("constraintRotatePivot")
	node : PointOnPolyConstraint = None
	pass
class ConstraintRotatePivotZPlug(Plug):
	parent : ConstraintRotatePivotPlug = PlugDescriptor("constraintRotatePivot")
	node : PointOnPolyConstraint = None
	pass
class ConstraintRotatePivotPlug(Plug):
	constraintRotatePivotX_ : ConstraintRotatePivotXPlug = PlugDescriptor("constraintRotatePivotX")
	crpx_ : ConstraintRotatePivotXPlug = PlugDescriptor("constraintRotatePivotX")
	constraintRotatePivotY_ : ConstraintRotatePivotYPlug = PlugDescriptor("constraintRotatePivotY")
	crpy_ : ConstraintRotatePivotYPlug = PlugDescriptor("constraintRotatePivotY")
	constraintRotatePivotZ_ : ConstraintRotatePivotZPlug = PlugDescriptor("constraintRotatePivotZ")
	crpz_ : ConstraintRotatePivotZPlug = PlugDescriptor("constraintRotatePivotZ")
	node : PointOnPolyConstraint = None
	pass
class ConstraintRotateTranslateXPlug(Plug):
	parent : ConstraintRotateTranslatePlug = PlugDescriptor("constraintRotateTranslate")
	node : PointOnPolyConstraint = None
	pass
class ConstraintRotateTranslateYPlug(Plug):
	parent : ConstraintRotateTranslatePlug = PlugDescriptor("constraintRotateTranslate")
	node : PointOnPolyConstraint = None
	pass
class ConstraintRotateTranslateZPlug(Plug):
	parent : ConstraintRotateTranslatePlug = PlugDescriptor("constraintRotateTranslate")
	node : PointOnPolyConstraint = None
	pass
class ConstraintRotateTranslatePlug(Plug):
	constraintRotateTranslateX_ : ConstraintRotateTranslateXPlug = PlugDescriptor("constraintRotateTranslateX")
	crtx_ : ConstraintRotateTranslateXPlug = PlugDescriptor("constraintRotateTranslateX")
	constraintRotateTranslateY_ : ConstraintRotateTranslateYPlug = PlugDescriptor("constraintRotateTranslateY")
	crty_ : ConstraintRotateTranslateYPlug = PlugDescriptor("constraintRotateTranslateY")
	constraintRotateTranslateZ_ : ConstraintRotateTranslateZPlug = PlugDescriptor("constraintRotateTranslateZ")
	crtz_ : ConstraintRotateTranslateZPlug = PlugDescriptor("constraintRotateTranslateZ")
	node : PointOnPolyConstraint = None
	pass
class ConstraintTranslateXPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : PointOnPolyConstraint = None
	pass
class ConstraintTranslateYPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : PointOnPolyConstraint = None
	pass
class ConstraintTranslateZPlug(Plug):
	parent : ConstraintTranslatePlug = PlugDescriptor("constraintTranslate")
	node : PointOnPolyConstraint = None
	pass
class ConstraintTranslatePlug(Plug):
	constraintTranslateX_ : ConstraintTranslateXPlug = PlugDescriptor("constraintTranslateX")
	ctx_ : ConstraintTranslateXPlug = PlugDescriptor("constraintTranslateX")
	constraintTranslateY_ : ConstraintTranslateYPlug = PlugDescriptor("constraintTranslateY")
	cty_ : ConstraintTranslateYPlug = PlugDescriptor("constraintTranslateY")
	constraintTranslateZ_ : ConstraintTranslateZPlug = PlugDescriptor("constraintTranslateZ")
	ctz_ : ConstraintTranslateZPlug = PlugDescriptor("constraintTranslateZ")
	node : PointOnPolyConstraint = None
	pass
class OffsetRotateXPlug(Plug):
	parent : OffsetRotatePlug = PlugDescriptor("offsetRotate")
	node : PointOnPolyConstraint = None
	pass
class OffsetRotateYPlug(Plug):
	parent : OffsetRotatePlug = PlugDescriptor("offsetRotate")
	node : PointOnPolyConstraint = None
	pass
class OffsetRotateZPlug(Plug):
	parent : OffsetRotatePlug = PlugDescriptor("offsetRotate")
	node : PointOnPolyConstraint = None
	pass
class OffsetRotatePlug(Plug):
	offsetRotateX_ : OffsetRotateXPlug = PlugDescriptor("offsetRotateX")
	orx_ : OffsetRotateXPlug = PlugDescriptor("offsetRotateX")
	offsetRotateY_ : OffsetRotateYPlug = PlugDescriptor("offsetRotateY")
	ory_ : OffsetRotateYPlug = PlugDescriptor("offsetRotateY")
	offsetRotateZ_ : OffsetRotateZPlug = PlugDescriptor("offsetRotateZ")
	orz_ : OffsetRotateZPlug = PlugDescriptor("offsetRotateZ")
	node : PointOnPolyConstraint = None
	pass
class OffsetTranslateXPlug(Plug):
	parent : OffsetTranslatePlug = PlugDescriptor("offsetTranslate")
	node : PointOnPolyConstraint = None
	pass
class OffsetTranslateYPlug(Plug):
	parent : OffsetTranslatePlug = PlugDescriptor("offsetTranslate")
	node : PointOnPolyConstraint = None
	pass
class OffsetTranslateZPlug(Plug):
	parent : OffsetTranslatePlug = PlugDescriptor("offsetTranslate")
	node : PointOnPolyConstraint = None
	pass
class OffsetTranslatePlug(Plug):
	offsetTranslateX_ : OffsetTranslateXPlug = PlugDescriptor("offsetTranslateX")
	otx_ : OffsetTranslateXPlug = PlugDescriptor("offsetTranslateX")
	offsetTranslateY_ : OffsetTranslateYPlug = PlugDescriptor("offsetTranslateY")
	oty_ : OffsetTranslateYPlug = PlugDescriptor("offsetTranslateY")
	offsetTranslateZ_ : OffsetTranslateZPlug = PlugDescriptor("offsetTranslateZ")
	otz_ : OffsetTranslateZPlug = PlugDescriptor("offsetTranslateZ")
	node : PointOnPolyConstraint = None
	pass
class RestRotateXPlug(Plug):
	parent : RestRotatePlug = PlugDescriptor("restRotate")
	node : PointOnPolyConstraint = None
	pass
class RestRotateYPlug(Plug):
	parent : RestRotatePlug = PlugDescriptor("restRotate")
	node : PointOnPolyConstraint = None
	pass
class RestRotateZPlug(Plug):
	parent : RestRotatePlug = PlugDescriptor("restRotate")
	node : PointOnPolyConstraint = None
	pass
class RestRotatePlug(Plug):
	restRotateX_ : RestRotateXPlug = PlugDescriptor("restRotateX")
	rrx_ : RestRotateXPlug = PlugDescriptor("restRotateX")
	restRotateY_ : RestRotateYPlug = PlugDescriptor("restRotateY")
	rry_ : RestRotateYPlug = PlugDescriptor("restRotateY")
	restRotateZ_ : RestRotateZPlug = PlugDescriptor("restRotateZ")
	rrz_ : RestRotateZPlug = PlugDescriptor("restRotateZ")
	node : PointOnPolyConstraint = None
	pass
class RestTranslateXPlug(Plug):
	parent : RestTranslatePlug = PlugDescriptor("restTranslate")
	node : PointOnPolyConstraint = None
	pass
class RestTranslateYPlug(Plug):
	parent : RestTranslatePlug = PlugDescriptor("restTranslate")
	node : PointOnPolyConstraint = None
	pass
class RestTranslateZPlug(Plug):
	parent : RestTranslatePlug = PlugDescriptor("restTranslate")
	node : PointOnPolyConstraint = None
	pass
class RestTranslatePlug(Plug):
	restTranslateX_ : RestTranslateXPlug = PlugDescriptor("restTranslateX")
	rtx_ : RestTranslateXPlug = PlugDescriptor("restTranslateX")
	restTranslateY_ : RestTranslateYPlug = PlugDescriptor("restTranslateY")
	rty_ : RestTranslateYPlug = PlugDescriptor("restTranslateY")
	restTranslateZ_ : RestTranslateZPlug = PlugDescriptor("restTranslateZ")
	rtz_ : RestTranslateZPlug = PlugDescriptor("restTranslateZ")
	node : PointOnPolyConstraint = None
	pass
class TargetMeshPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : PointOnPolyConstraint = None
	pass
class TargetVPlug(Plug):
	parent : TargetUVPlug = PlugDescriptor("targetUV")
	node : PointOnPolyConstraint = None
	pass
class TargetUVPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetU_ : TargetUPlug = PlugDescriptor("targetU")
	tu_ : TargetUPlug = PlugDescriptor("targetU")
	targetV_ : TargetVPlug = PlugDescriptor("targetV")
	tv_ : TargetVPlug = PlugDescriptor("targetV")
	node : PointOnPolyConstraint = None
	pass
class TargetUVSetNamePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : PointOnPolyConstraint = None
	pass
class TargetUseNormalPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : PointOnPolyConstraint = None
	pass
class TargetWeightPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : PointOnPolyConstraint = None
	pass
class TargetPlug(Plug):
	targetMesh_ : TargetMeshPlug = PlugDescriptor("targetMesh")
	tm_ : TargetMeshPlug = PlugDescriptor("targetMesh")
	targetUV_ : TargetUVPlug = PlugDescriptor("targetUV")
	tuv_ : TargetUVPlug = PlugDescriptor("targetUV")
	targetUVSetName_ : TargetUVSetNamePlug = PlugDescriptor("targetUVSetName")
	tnm_ : TargetUVSetNamePlug = PlugDescriptor("targetUVSetName")
	targetUseNormal_ : TargetUseNormalPlug = PlugDescriptor("targetUseNormal")
	tun_ : TargetUseNormalPlug = PlugDescriptor("targetUseNormal")
	targetWeight_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	tw_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	node : PointOnPolyConstraint = None
	pass
class TargetUPlug(Plug):
	parent : TargetUVPlug = PlugDescriptor("targetUV")
	node : PointOnPolyConstraint = None
	pass
# endregion


# define node class
class PointOnPolyConstraint(Constraint):
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
	offsetRotateX_ : OffsetRotateXPlug = PlugDescriptor("offsetRotateX")
	offsetRotateY_ : OffsetRotateYPlug = PlugDescriptor("offsetRotateY")
	offsetRotateZ_ : OffsetRotateZPlug = PlugDescriptor("offsetRotateZ")
	offsetRotate_ : OffsetRotatePlug = PlugDescriptor("offsetRotate")
	offsetTranslateX_ : OffsetTranslateXPlug = PlugDescriptor("offsetTranslateX")
	offsetTranslateY_ : OffsetTranslateYPlug = PlugDescriptor("offsetTranslateY")
	offsetTranslateZ_ : OffsetTranslateZPlug = PlugDescriptor("offsetTranslateZ")
	offsetTranslate_ : OffsetTranslatePlug = PlugDescriptor("offsetTranslate")
	restRotateX_ : RestRotateXPlug = PlugDescriptor("restRotateX")
	restRotateY_ : RestRotateYPlug = PlugDescriptor("restRotateY")
	restRotateZ_ : RestRotateZPlug = PlugDescriptor("restRotateZ")
	restRotate_ : RestRotatePlug = PlugDescriptor("restRotate")
	restTranslateX_ : RestTranslateXPlug = PlugDescriptor("restTranslateX")
	restTranslateY_ : RestTranslateYPlug = PlugDescriptor("restTranslateY")
	restTranslateZ_ : RestTranslateZPlug = PlugDescriptor("restTranslateZ")
	restTranslate_ : RestTranslatePlug = PlugDescriptor("restTranslate")
	targetMesh_ : TargetMeshPlug = PlugDescriptor("targetMesh")
	targetV_ : TargetVPlug = PlugDescriptor("targetV")
	targetUV_ : TargetUVPlug = PlugDescriptor("targetUV")
	targetUVSetName_ : TargetUVSetNamePlug = PlugDescriptor("targetUVSetName")
	targetUseNormal_ : TargetUseNormalPlug = PlugDescriptor("targetUseNormal")
	targetWeight_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	target_ : TargetPlug = PlugDescriptor("target")
	targetU_ : TargetUPlug = PlugDescriptor("targetU")

	# node attributes

	typeName = "pointOnPolyConstraint"
	apiTypeInt = 1060
	apiTypeStr = "kPointOnPolyConstraint"
	typeIdInt = 1146114115
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["constraintParentInverseMatrix", "constraintRotateX", "constraintRotateY", "constraintRotateZ", "constraintRotate", "constraintRotateOrder", "constraintRotatePivotX", "constraintRotatePivotY", "constraintRotatePivotZ", "constraintRotatePivot", "constraintRotateTranslateX", "constraintRotateTranslateY", "constraintRotateTranslateZ", "constraintRotateTranslate", "constraintTranslateX", "constraintTranslateY", "constraintTranslateZ", "constraintTranslate", "offsetRotateX", "offsetRotateY", "offsetRotateZ", "offsetRotate", "offsetTranslateX", "offsetTranslateY", "offsetTranslateZ", "offsetTranslate", "restRotateX", "restRotateY", "restRotateZ", "restRotate", "restTranslateX", "restTranslateY", "restTranslateZ", "restTranslate", "targetMesh", "targetV", "targetUV", "targetUVSetName", "targetUseNormal", "targetWeight", "target", "targetU"]
	nodeLeafPlugs = ["constraintParentInverseMatrix", "constraintRotate", "constraintRotateOrder", "constraintRotatePivot", "constraintRotateTranslate", "constraintTranslate", "offsetRotate", "offsetTranslate", "restRotate", "restTranslate", "target"]
	pass

