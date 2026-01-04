

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
class ConstraintGeometryPlug(Plug):
	node : GeometryConstraint = None
	pass
class ConstraintParentInverseMatrixPlug(Plug):
	node : GeometryConstraint = None
	pass
class TargetGeometryPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : GeometryConstraint = None
	pass
class TargetWeightPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : GeometryConstraint = None
	pass
class TargetPlug(Plug):
	targetGeometry_ : TargetGeometryPlug = PlugDescriptor("targetGeometry")
	tgm_ : TargetGeometryPlug = PlugDescriptor("targetGeometry")
	targetWeight_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	tw_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	node : GeometryConstraint = None
	pass
# endregion


# define node class
class GeometryConstraint(Constraint):
	constraintGeometry_ : ConstraintGeometryPlug = PlugDescriptor("constraintGeometry")
	constraintParentInverseMatrix_ : ConstraintParentInverseMatrixPlug = PlugDescriptor("constraintParentInverseMatrix")
	targetGeometry_ : TargetGeometryPlug = PlugDescriptor("targetGeometry")
	targetWeight_ : TargetWeightPlug = PlugDescriptor("targetWeight")
	target_ : TargetPlug = PlugDescriptor("target")

	# node attributes

	typeName = "geometryConstraint"
	apiTypeInt = 113
	apiTypeStr = "kGeometryConstraint"
	typeIdInt = 1145523779
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["constraintGeometry", "constraintParentInverseMatrix", "targetGeometry", "targetWeight", "target"]
	nodeLeafPlugs = ["constraintGeometry", "constraintParentInverseMatrix", "target"]
	pass

