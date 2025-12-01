

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
class InputSurfacePlug(Plug):
	node : CurveFromSurface = None
	pass
class MaxValuePlug(Plug):
	node : CurveFromSurface = None
	pass
class MinValuePlug(Plug):
	node : CurveFromSurface = None
	pass
class OutputCurvePlug(Plug):
	node : CurveFromSurface = None
	pass
class RelativePlug(Plug):
	node : CurveFromSurface = None
	pass
# endregion


# define node class
class CurveFromSurface(CurveRange):
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	maxValue_ : MaxValuePlug = PlugDescriptor("maxValue")
	minValue_ : MinValuePlug = PlugDescriptor("minValue")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	relative_ : RelativePlug = PlugDescriptor("relative")

	# node attributes

	typeName = "curveFromSurface"
	typeIdInt = 1313031763
	nodeLeafClassAttrs = ["inputSurface", "maxValue", "minValue", "outputCurve", "relative"]
	nodeLeafPlugs = ["inputSurface", "maxValue", "minValue", "outputCurve", "relative"]
	pass

