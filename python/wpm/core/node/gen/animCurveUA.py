

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
	node : AnimCurveUA = None
	pass
class KeyTimePlug(Plug):
	parent : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	node : AnimCurveUA = None
	pass
class KeyValuePlug(Plug):
	parent : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	node : AnimCurveUA = None
	pass
class KeyTimeValuePlug(Plug):
	keyTime_ : KeyTimePlug = PlugDescriptor("keyTime")
	kt_ : KeyTimePlug = PlugDescriptor("keyTime")
	keyValue_ : KeyValuePlug = PlugDescriptor("keyValue")
	kv_ : KeyValuePlug = PlugDescriptor("keyValue")
	node : AnimCurveUA = None
	pass
class OutputPlug(Plug):
	node : AnimCurveUA = None
	pass
# endregion


# define node class
class AnimCurveUA(AnimCurve):
	input_ : InputPlug = PlugDescriptor("input")
	keyTime_ : KeyTimePlug = PlugDescriptor("keyTime")
	keyValue_ : KeyValuePlug = PlugDescriptor("keyValue")
	keyTimeValue_ : KeyTimeValuePlug = PlugDescriptor("keyTimeValue")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "animCurveUA"
	typeIdInt = 1346590017
	nodeLeafClassAttrs = ["input", "keyTime", "keyValue", "keyTimeValue", "output"]
	nodeLeafPlugs = ["input", "keyTimeValue", "output"]
	pass

