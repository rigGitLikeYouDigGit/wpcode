

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
	node : AnimCurveTT = None
	pass
class KeyTimePlug(Plug):
	parent : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	node : AnimCurveTT = None
	pass
class KeyValuePlug(Plug):
	parent : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	node : AnimCurveTT = None
	pass
class KeyTimeValuePlug(Plug):
	keyTime_ : KeyTimePlug = PlugDescriptor("keyTime")
	kt_ : KeyTimePlug = PlugDescriptor("keyTime")
	keyValue_ : KeyValuePlug = PlugDescriptor("keyValue")
	kv_ : KeyValuePlug = PlugDescriptor("keyValue")
	node : AnimCurveTT = None
	pass
class KeyValueSwitchPlug(Plug):
	node : AnimCurveTT = None
	pass
class OutputPlug(Plug):
	node : AnimCurveTT = None
	pass
# endregion


# define node class
class AnimCurveTT(AnimCurve):
	input_ : InputPlug = PlugDescriptor("input")
	keyTime_ : KeyTimePlug = PlugDescriptor("keyTime")
	keyValue_ : KeyValuePlug = PlugDescriptor("keyValue")
	keyTimeValue_ : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	keyValueSwitch_ : KeyValueSwitchPlug = PlugDescriptor("keyValueSwitch")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "animCurveTT"
	typeIdInt = 1346589780
	nodeLeafClassAttrs = ["input", "keyTime", "keyValue", "keyTimeValue", "keyValueSwitch", "output"]
	nodeLeafPlugs = ["input", "keyTimeValue", "keyValueSwitch", "output"]
	pass

