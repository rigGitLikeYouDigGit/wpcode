

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
	node : ResultCurveTimeToAngular = None
	pass
class InputResultPlug(Plug):
	node : ResultCurveTimeToAngular = None
	pass
class KeyTimePlug(Plug):
	parent : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	node : ResultCurveTimeToAngular = None
	pass
class KeyValuePlug(Plug):
	parent : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	node : ResultCurveTimeToAngular = None
	pass
class KeyTimeValuePlug(Plug):
	keyTime_ : KeyTimePlug = PlugDescriptor("keyTime")
	kt_ : KeyTimePlug = PlugDescriptor("keyTime")
	keyValue_ : KeyValuePlug = PlugDescriptor("keyValue")
	kv_ : KeyValuePlug = PlugDescriptor("keyValue")
	node : ResultCurveTimeToAngular = None
	pass
class OutputPlug(Plug):
	node : ResultCurveTimeToAngular = None
	pass
# endregion


# define node class
class ResultCurveTimeToAngular(ResultCurve):
	input_ : InputPlug = PlugDescriptor("input")
	inputResult_ : InputResultPlug = PlugDescriptor("inputResult")
	keyTime_ : KeyTimePlug = PlugDescriptor("keyTime")
	keyValue_ : KeyValuePlug = PlugDescriptor("keyValue")
	keyTimeValue_ : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "resultCurveTimeToAngular"
	apiTypeInt = 17
	apiTypeStr = "kResultCurveTimeToAngular"
	typeIdInt = 1380144193
	MFnCls = om.MFnAnimCurve
	pass

