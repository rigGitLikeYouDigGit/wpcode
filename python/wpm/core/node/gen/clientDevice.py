

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ThreadedDevice = retriever.getNodeCls("ThreadedDevice")
assert ThreadedDevice
if T.TYPE_CHECKING:
	from .. import ThreadedDevice

# add node doc



# region plug type defs
class DeviceNamePlug(Plug):
	node : ClientDevice = None
	pass
class ServerNamePlug(Plug):
	node : ClientDevice = None
	pass
# endregion


# define node class
class ClientDevice(ThreadedDevice):
	deviceName_ : DeviceNamePlug = PlugDescriptor("deviceName")
	serverName_ : ServerNamePlug = PlugDescriptor("serverName")

	# node attributes

	typeName = "clientDevice"
	apiTypeInt = 1077
	apiTypeStr = "kClientDevice"
	typeIdInt = 1668047990
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["deviceName", "serverName"]
	nodeLeafPlugs = ["deviceName", "serverName"]
	pass

