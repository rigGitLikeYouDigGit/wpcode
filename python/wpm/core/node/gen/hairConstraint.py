

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Shape = retriever.getNodeCls("Shape")
assert Shape
if T.TYPE_CHECKING:
	from .. import Shape

# add node doc



# region plug type defs
class ConstraintMethodPlug(Plug):
	node : HairConstraint = None
	pass
class CurveIndicesPlug(Plug):
	node : HairConstraint = None
	pass
class GlueStrengthPlug(Plug):
	node : HairConstraint = None
	pass
class OutPinPlug(Plug):
	node : HairConstraint = None
	pass
class PointMethodPlug(Plug):
	node : HairConstraint = None
	pass
class StiffnessPlug(Plug):
	node : HairConstraint = None
	pass
class UDistancePlug(Plug):
	node : HairConstraint = None
	pass
class UParameterPlug(Plug):
	node : HairConstraint = None
	pass
# endregion


# define node class
class HairConstraint(Shape):
	constraintMethod_ : ConstraintMethodPlug = PlugDescriptor("constraintMethod")
	curveIndices_ : CurveIndicesPlug = PlugDescriptor("curveIndices")
	glueStrength_ : GlueStrengthPlug = PlugDescriptor("glueStrength")
	outPin_ : OutPinPlug = PlugDescriptor("outPin")
	pointMethod_ : PointMethodPlug = PlugDescriptor("pointMethod")
	stiffness_ : StiffnessPlug = PlugDescriptor("stiffness")
	uDistance_ : UDistancePlug = PlugDescriptor("uDistance")
	uParameter_ : UParameterPlug = PlugDescriptor("uParameter")

	# node attributes

	typeName = "hairConstraint"
	apiTypeInt = 940
	apiTypeStr = "kHairConstraint"
	typeIdInt = 1213221198
	MFnCls = om.MFnDagNode
	pass

