

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
class InputSubdivPlug(Plug):
	node : CurveFromSubdiv = None
	pass
class MaxValuePlug(Plug):
	node : CurveFromSubdiv = None
	pass
class MinValuePlug(Plug):
	node : CurveFromSubdiv = None
	pass
class OutputCurvePlug(Plug):
	node : CurveFromSubdiv = None
	pass
class RelativePlug(Plug):
	node : CurveFromSubdiv = None
	pass
# endregion


# define node class
class CurveFromSubdiv(CurveRange):
	inputSubdiv_ : InputSubdivPlug = PlugDescriptor("inputSubdiv")
	maxValue_ : MaxValuePlug = PlugDescriptor("maxValue")
	minValue_ : MinValuePlug = PlugDescriptor("minValue")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	relative_ : RelativePlug = PlugDescriptor("relative")

	# node attributes

	typeName = "curveFromSubdiv"
	typeIdInt = 1396917843
	pass

