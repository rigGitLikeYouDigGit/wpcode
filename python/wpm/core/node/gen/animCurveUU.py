

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AnimCurve = retriever.getNodeCls("AnimCurve")
assert AnimCurve
if T.TYPE_CHECKING:
	from .. import AnimCurve

# add node doc



# region plug type defs
class InputPlug(Plug):
	node : AnimCurveUU = None
	pass
class KeyTimePlug(Plug):
	parent : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	node : AnimCurveUU = None
	pass
class KeyValuePlug(Plug):
	parent : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	node : AnimCurveUU = None
	pass
class KeyTimeValuePlug(Plug):
	keyTime_ : KeyTimePlug = PlugDescriptor("keyTime")
	kt_ : KeyTimePlug = PlugDescriptor("keyTime")
	keyValue_ : KeyValuePlug = PlugDescriptor("keyValue")
	kv_ : KeyValuePlug = PlugDescriptor("keyValue")
	node : AnimCurveUU = None
	pass
class OutputPlug(Plug):
	node : AnimCurveUU = None
	pass
# endregion


# define node class
class AnimCurveUU(AnimCurve):
	input_ : InputPlug = PlugDescriptor("input")
	keyTime_ : KeyTimePlug = PlugDescriptor("keyTime")
	keyValue_ : KeyValuePlug = PlugDescriptor("keyValue")
	keyTimeValue_ : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "animCurveUU"
	typeIdInt = 1346590037
	nodeLeafClassAttrs = ["input", "keyTime", "keyValue", "keyTimeValue", "output"]
	nodeLeafPlugs = ["input", "keyTimeValue", "output"]
	pass

