

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	AnimCurve = Catalogue.AnimCurve
else:
	from .. import retriever
	AnimCurve = retriever.getNodeCls("AnimCurve")
	assert AnimCurve

# add node doc



# region plug type defs
class InputPlug(Plug):
	node : AnimCurveTL = None
	pass
class KeyTimePlug(Plug):
	parent : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	node : AnimCurveTL = None
	pass
class KeyValuePlug(Plug):
	parent : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	node : AnimCurveTL = None
	pass
class KeyTimeValuePlug(Plug):
	keyTime_ : KeyTimePlug = PlugDescriptor("keyTime")
	kt_ : KeyTimePlug = PlugDescriptor("keyTime")
	keyValue_ : KeyValuePlug = PlugDescriptor("keyValue")
	kv_ : KeyValuePlug = PlugDescriptor("keyValue")
	node : AnimCurveTL = None
	pass
class KeyValueSwitchPlug(Plug):
	node : AnimCurveTL = None
	pass
class OutputPlug(Plug):
	node : AnimCurveTL = None
	pass
# endregion


# define node class
class AnimCurveTL(AnimCurve):
	input_ : InputPlug = PlugDescriptor("input")
	keyTime_ : KeyTimePlug = PlugDescriptor("keyTime")
	keyValue_ : KeyValuePlug = PlugDescriptor("keyValue")
	keyTimeValue_ : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	keyValueSwitch_ : KeyValueSwitchPlug = PlugDescriptor("keyValueSwitch")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "animCurveTL"
	typeIdInt = 1346589772
	nodeLeafClassAttrs = ["input", "keyTime", "keyValue", "keyTimeValue", "keyValueSwitch", "output"]
	nodeLeafPlugs = ["input", "keyTimeValue", "keyValueSwitch", "output"]
	pass

