

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ResultCurve = Catalogue.ResultCurve
else:
	from .. import retriever
	ResultCurve = retriever.getNodeCls("ResultCurve")
	assert ResultCurve

# add node doc



# region plug type defs
class InputPlug(Plug):
	node : ResultCurveTimeToUnitless = None
	pass
class InputResultPlug(Plug):
	node : ResultCurveTimeToUnitless = None
	pass
class KeyTimePlug(Plug):
	parent : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	node : ResultCurveTimeToUnitless = None
	pass
class KeyValuePlug(Plug):
	parent : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	node : ResultCurveTimeToUnitless = None
	pass
class KeyTimeValuePlug(Plug):
	keyTime_ : KeyTimePlug = PlugDescriptor("keyTime")
	kt_ : KeyTimePlug = PlugDescriptor("keyTime")
	keyValue_ : KeyValuePlug = PlugDescriptor("keyValue")
	kv_ : KeyValuePlug = PlugDescriptor("keyValue")
	node : ResultCurveTimeToUnitless = None
	pass
class OutputPlug(Plug):
	node : ResultCurveTimeToUnitless = None
	pass
# endregion


# define node class
class ResultCurveTimeToUnitless(ResultCurve):
	input_ : InputPlug = PlugDescriptor("input")
	inputResult_ : InputResultPlug = PlugDescriptor("inputResult")
	keyTime_ : KeyTimePlug = PlugDescriptor("keyTime")
	keyValue_ : KeyValuePlug = PlugDescriptor("keyValue")
	keyTimeValue_ : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "resultCurveTimeToUnitless"
	apiTypeInt = 20
	apiTypeStr = "kResultCurveTimeToUnitless"
	typeIdInt = 1380144213
	MFnCls = om.MFnAnimCurve
	nodeLeafClassAttrs = ["input", "inputResult", "keyTime", "keyValue", "keyTimeValue", "output"]
	nodeLeafPlugs = ["input", "inputResult", "keyTimeValue", "output"]
	pass

