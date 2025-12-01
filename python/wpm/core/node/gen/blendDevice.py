

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Blend = retriever.getNodeCls("Blend")
assert Blend
if T.TYPE_CHECKING:
	from .. import Blend

# add node doc



# region plug type defs
class BlenderPlug(Plug):
	node : BlendDevice = None
	pass
class DataPlug(Plug):
	node : BlendDevice = None
	pass
class DeviceBlenderPlug(Plug):
	node : BlendDevice = None
	pass
class DeviceValuePlug(Plug):
	node : BlendDevice = None
	pass
class InputAnglePlug(Plug):
	node : BlendDevice = None
	pass
class InputLinearPlug(Plug):
	node : BlendDevice = None
	pass
class MinTimePlug(Plug):
	node : BlendDevice = None
	pass
class OffsetPlug(Plug):
	node : BlendDevice = None
	pass
class OutputAnglePlug(Plug):
	node : BlendDevice = None
	pass
class OutputLinearPlug(Plug):
	node : BlendDevice = None
	pass
class PeriodPlug(Plug):
	node : BlendDevice = None
	pass
class StridePlug(Plug):
	node : BlendDevice = None
	pass
class TimePlug(Plug):
	node : BlendDevice = None
	pass
class TimeStampPlug(Plug):
	node : BlendDevice = None
	pass
# endregion


# define node class
class BlendDevice(Blend):
	blender_ : BlenderPlug = PlugDescriptor("blender")
	data_ : DataPlug = PlugDescriptor("data")
	deviceBlender_ : DeviceBlenderPlug = PlugDescriptor("deviceBlender")
	deviceValue_ : DeviceValuePlug = PlugDescriptor("deviceValue")
	inputAngle_ : InputAnglePlug = PlugDescriptor("inputAngle")
	inputLinear_ : InputLinearPlug = PlugDescriptor("inputLinear")
	minTime_ : MinTimePlug = PlugDescriptor("minTime")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	outputAngle_ : OutputAnglePlug = PlugDescriptor("outputAngle")
	outputLinear_ : OutputLinearPlug = PlugDescriptor("outputLinear")
	period_ : PeriodPlug = PlugDescriptor("period")
	stride_ : StridePlug = PlugDescriptor("stride")
	time_ : TimePlug = PlugDescriptor("time")
	timeStamp_ : TimeStampPlug = PlugDescriptor("timeStamp")

	# node attributes

	typeName = "blendDevice"
	apiTypeInt = 30
	apiTypeStr = "kBlendDevice"
	typeIdInt = 1112294486
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["blender", "data", "deviceBlender", "deviceValue", "inputAngle", "inputLinear", "minTime", "offset", "outputAngle", "outputLinear", "period", "stride", "time", "timeStamp"]
	nodeLeafPlugs = ["blender", "data", "deviceBlender", "deviceValue", "inputAngle", "inputLinear", "minTime", "offset", "outputAngle", "outputLinear", "period", "stride", "time", "timeStamp"]
	pass

