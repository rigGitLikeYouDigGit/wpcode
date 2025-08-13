

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ResultCurve = retriever.getNodeCls("ResultCurve")
assert ResultCurve
if T.TYPE_CHECKING:
	from .. import ResultCurve

# add node doc



# region plug type defs
class InputPlug(Plug):
	node : ResultCurveTimeToLinear = None
	pass
class InputResultPlug(Plug):
	node : ResultCurveTimeToLinear = None
	pass
class KeyTimePlug(Plug):
	parent : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	node : ResultCurveTimeToLinear = None
	pass
class KeyValuePlug(Plug):
	parent : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	node : ResultCurveTimeToLinear = None
	pass
class KeyTimeValuePlug(Plug):
	keyTime_ : KeyTimePlug = PlugDescriptor("keyTime")
	kt_ : KeyTimePlug = PlugDescriptor("keyTime")
	keyValue_ : KeyValuePlug = PlugDescriptor("keyValue")
	kv_ : KeyValuePlug = PlugDescriptor("keyValue")
	node : ResultCurveTimeToLinear = None
	pass
class OutputPlug(Plug):
	node : ResultCurveTimeToLinear = None
	pass
# endregion


# define node class
class ResultCurveTimeToLinear(ResultCurve):
	input_ : InputPlug = PlugDescriptor("input")
	inputResult_ : InputResultPlug = PlugDescriptor("inputResult")
	keyTime_ : KeyTimePlug = PlugDescriptor("keyTime")
	keyValue_ : KeyValuePlug = PlugDescriptor("keyValue")
	keyTimeValue_ : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "resultCurveTimeToLinear"
	typeIdInt = 1380144204
	pass

