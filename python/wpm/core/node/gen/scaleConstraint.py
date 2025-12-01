

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
class ConstraintParentInverseMatrixPlug(Plug):
	node : ScaleConstraint = None
	pass
class ConstraintScaleXPlug(Plug):
	parent : ConstraintScalePlug = PlugDescriptor("constraintScale")
	node : ScaleConstraint = None
	pass
class ConstraintScaleYPlug(Plug):
	parent : ConstraintScalePlug = PlugDescriptor("constraintScale")
	node : ScaleConstraint = None
	pass
class ConstraintScaleZPlug(Plug):
	parent : ConstraintScalePlug = PlugDescriptor("constraintScale")
	node : ScaleConstraint = None
	pass
class ConstraintScalePlug(Plug):
	constraintScaleX_ : ConstraintScaleXPlug = PlugDescriptor("constraintScaleX")
	csx_ : ConstraintScaleXPlug = PlugDescriptor("constraintScaleX")
	constraintScaleY_ : ConstraintScaleYPlug = PlugDescriptor("constraintScaleY")
	csy_ : ConstraintScaleYPlug = PlugDescriptor("constraintScaleY")
	constraintScaleZ_ : ConstraintScaleZPlug = PlugDescriptor("constraintScaleZ")
	csz_ : ConstraintScaleZPlug = PlugDescriptor("constraintScaleZ")
	node : ScaleConstraint = None
	pass
class ConstraintScaleCompensatePlug(Plug):
	node : ScaleConstraint = None
	pass
class OffsetXPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : ScaleConstraint = None
	pass
class OffsetYPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : ScaleConstraint = None
	pass
class OffsetZPlug(Plug):
	parent : OffsetPlug = PlugDescriptor("offset")
	node : ScaleConstraint = None
	pass
class OffsetPlug(Plug):
	offsetX_ : OffsetXPlug = PlugDescriptor("offsetX")
	ox_ : OffsetXPlug = PlugDescriptor("offsetX")
	offsetY_ : OffsetYPlug = PlugDescriptor("offsetY")
	oy_ : OffsetYPlug = PlugDescriptor("offsetY")
	offsetZ_ : OffsetZPlug = PlugDescriptor("offsetZ")
	oz_ : OffsetZPlug = PlugDescriptor("offsetZ")
	node : ScaleConstraint = None
	pass
class RestScaleXPlug(Plug):
	parent : RestScalePlug = PlugDescriptor("restScale")
	node : ScaleConstraint = None
	pass
class RestScaleYPlug(Plug):
	parent : RestScalePlug = PlugDescriptor("restScale")
	node : ScaleConstraint = None
	pass
class RestScaleZPlug(Plug):
	parent : RestScalePlug = PlugDescriptor("restScale")
	node : ScaleConstraint = None
	pass
class RestScalePlug(Plug):
	restScaleX_ : RestScaleXPlug = PlugDescriptor("restScaleX")
	rsx_ : RestScaleXPlug = PlugDescriptor("restScaleX")
	restScaleY_ : RestScaleYPlug = PlugDescriptor("restScaleY")
	rsy_ : RestScaleYPlug = PlugDescriptor("restScaleY")
	restScaleZ_ : RestScaleZPlug = PlugDescriptor("restScaleZ")
	rsz_ : RestScaleZPlug = PlugDescriptor("restScaleZ")
	node : ScaleConstraint = None
	pass
class TargetParentMatrixPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : ScaleConstraint = None
	pass
class TargetScaleXPlug(Plug):
	parent : TargetScalePlug = PlugDescriptor("targetScale")
	node : ScaleConstraint = None
	pass
class TargetScaleYPlug(Plug):
	parent : TargetScalePlug = PlugDescriptor("targetScale")
	node : ScaleConstraint = None
	pass
class TargetScaleZPlug(Plug):
	parent : TargetScalePlug = PlugDescriptor("targetScale")
	node : ScaleConstraint = None
	pass
class TargetScalePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	targetScaleX_ : TargetScaleXPlug = PlugDescriptor("targetScaleX")
	tsx_ : TargetScaleXPlug = PlugDescriptor("targetScaleX")
	targetScaleY_ : TargetScaleYPlug = PlugDescriptor("targetScaleY")
	tsy_ : TargetScaleYPlug = PlugDescriptor("targetScaleY")
	targetScaleZ_ : TargetScaleZPlug = PlugDescriptor("targetScaleZ")
	tsz_ : TargetScaleZPlug = PlugDescriptor("targetScaleZ")
	node : ScaleConstraint = None
	pass
class TargetWeightPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : ScaleConstraint = None
	pass
class TargetPlug(Plug):
	targetParentMatrix_ : TargetParentMatrixPlug = PlugDescriptor("targetParentMatrix")
	tpm_ : TargetParentMatrixPlug = PlugDescriptor("targetParentMatrix")
	targetScale_ : TargetScalePlug = PlugDescriptor("targetScale")
	ts_ : TargetScalePlug = PlugDescriptor("targetScale")
	targetWeight_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	tw_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	node : ScaleConstraint = None
	pass
# endregion


# define node class
class ScaleConstraint(Constraint):
	constraintParentInverseMatrix_ : ConstraintParentInverseMatrixPlug = PlugDescriptor("constraintParentInverseMatrix")
	constraintScaleX_ : ConstraintScaleXPlug = PlugDescriptor("constraintScaleX")
	constraintScaleY_ : ConstraintScaleYPlug = PlugDescriptor("constraintScaleY")
	constraintScaleZ_ : ConstraintScaleZPlug = PlugDescriptor("constraintScaleZ")
	constraintScale_ : ConstraintScalePlug = PlugDescriptor("constraintScale")
	constraintScaleCompensate_ : ConstraintScaleCompensatePlug = PlugDescriptor("constraintScaleCompensate")
	offsetX_ : OffsetXPlug = PlugDescriptor("offsetX")
	offsetY_ : OffsetYPlug = PlugDescriptor("offsetY")
	offsetZ_ : OffsetZPlug = PlugDescriptor("offsetZ")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	restScaleX_ : RestScaleXPlug = PlugDescriptor("restScaleX")
	restScaleY_ : RestScaleYPlug = PlugDescriptor("restScaleY")
	restScaleZ_ : RestScaleZPlug = PlugDescriptor("restScaleZ")
	restScale_ : RestScalePlug = PlugDescriptor("restScale")
	targetParentMatrix_ : TargetParentMatrixPlug = PlugDescriptor("targetParentMatrix")
	targetScaleX_ : TargetScaleXPlug = PlugDescriptor("targetScaleX")
	targetScaleY_ : TargetScaleYPlug = PlugDescriptor("targetScaleY")
	targetScaleZ_ : TargetScaleZPlug = PlugDescriptor("targetScaleZ")
	targetScale_ : TargetScalePlug = PlugDescriptor("targetScale")
	targetWeight_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	target_ : TargetPlug = PlugDescriptor("target")

	# node attributes

	typeName = "scaleConstraint"
	apiTypeInt = 244
	apiTypeStr = "kScaleConstraint"
	typeIdInt = 1146307395
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["constraintParentInverseMatrix", "constraintScaleX", "constraintScaleY", "constraintScaleZ", "constraintScale", "constraintScaleCompensate", "offsetX", "offsetY", "offsetZ", "offset", "restScaleX", "restScaleY", "restScaleZ", "restScale", "targetParentMatrix", "targetScaleX", "targetScaleY", "targetScaleZ", "targetScale", "targetWeight", "target"]
	nodeLeafPlugs = ["constraintParentInverseMatrix", "constraintScale", "constraintScaleCompensate", "offset", "restScale", "target"]
	pass

