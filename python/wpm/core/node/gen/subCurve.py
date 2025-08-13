

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
CurveRange = retriever.getNodeCls("CurveRange")
assert CurveRange
if T.TYPE_CHECKING:
	from .. import CurveRange

# add node doc



# region plug type defs
class InputCurvePlug(Plug):
	node : SubCurve = None
	pass
class MaxValuePlug(Plug):
	node : SubCurve = None
	pass
class MinValuePlug(Plug):
	node : SubCurve = None
	pass
class OutputCurvePlug(Plug):
	node : SubCurve = None
	pass
class RelativePlug(Plug):
	node : SubCurve = None
	pass
# endregion


# define node class
class SubCurve(CurveRange):
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	maxValue_ : MaxValuePlug = PlugDescriptor("maxValue")
	minValue_ : MinValuePlug = PlugDescriptor("minValue")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	relative_ : RelativePlug = PlugDescriptor("relative")

	# node attributes

	typeName = "subCurve"
	apiTypeInt = 102
	apiTypeStr = "kSubCurve"
	typeIdInt = 1314079299
	MFnCls = om.MFnDependencyNode
	pass

