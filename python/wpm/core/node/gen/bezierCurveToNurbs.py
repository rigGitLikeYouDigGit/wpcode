

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
assert AbstractBaseCreate
if T.TYPE_CHECKING:
	from .. import AbstractBaseCreate

# add node doc



# region plug type defs
class InputCurvePlug(Plug):
	node : BezierCurveToNurbs = None
	pass
class OutputCurvePlug(Plug):
	node : BezierCurveToNurbs = None
	pass
# endregion


# define node class
class BezierCurveToNurbs(AbstractBaseCreate):
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")

	# node attributes

	typeName = "bezierCurveToNurbs"
	apiTypeInt = 1057
	apiTypeStr = "kBezierCurveToNurbs"
	typeIdInt = 1112821330
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["inputCurve", "outputCurve"]
	nodeLeafPlugs = ["inputCurve", "outputCurve"]
	pass

