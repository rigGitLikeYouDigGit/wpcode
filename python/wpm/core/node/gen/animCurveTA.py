

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
	node : AnimCurveTA = None
	pass
class KeyTimePlug(Plug):
	parent : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	node : AnimCurveTA = None
	pass
class KeyValuePlug(Plug):
	parent : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	node : AnimCurveTA = None
	pass
class KeyTimeValuePlug(Plug):
	keyTime_ : KeyTimePlug = PlugDescriptor("keyTime")
	kt_ : KeyTimePlug = PlugDescriptor("keyTime")
	keyValue_ : KeyValuePlug = PlugDescriptor("keyValue")
	kv_ : KeyValuePlug = PlugDescriptor("keyValue")
	node : AnimCurveTA = None
	pass
class KeyValueSwitchPlug(Plug):
	node : AnimCurveTA = None
	pass
class OutputPlug(Plug):
	node : AnimCurveTA = None
	pass
class QuaternionWPlug(Plug):
	node : AnimCurveTA = None
	pass
class RawValuePlug(Plug):
	node : AnimCurveTA = None
	pass
# endregion


# define node class
class AnimCurveTA(AnimCurve):
	input_ : InputPlug = PlugDescriptor("input")
	keyTime_ : KeyTimePlug = PlugDescriptor("keyTime")
	keyValue_ : KeyValuePlug = PlugDescriptor("keyValue")
	keyTimeValue_ : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	keyValueSwitch_ : KeyValueSwitchPlug = PlugDescriptor("keyValueSwitch")
	output_ : OutputPlug = PlugDescriptor("output")
	quaternionW_ : QuaternionWPlug = PlugDescriptor("quaternionW")
	rawValue_ : RawValuePlug = PlugDescriptor("rawValue")

	# node attributes

	typeName = "animCurveTA"
	typeIdInt = 1346589761
	nodeLeafClassAttrs = ["input", "keyTime", "keyValue", "keyTimeValue", "keyValueSwitch", "output", "quaternionW", "rawValue"]
	nodeLeafPlugs = ["input", "keyTimeValue", "keyValueSwitch", "output", "quaternionW", "rawValue"]
	pass

