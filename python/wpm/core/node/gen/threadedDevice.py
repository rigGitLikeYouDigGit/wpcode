

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
_BASE_ = retriever.getNodeCls("_BASE_")
assert _BASE_
if T.TYPE_CHECKING:
	from .. import _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : ThreadedDevice = None
	pass
class FrameRatePlug(Plug):
	node : ThreadedDevice = None
	pass
class LivePlug(Plug):
	node : ThreadedDevice = None
	pass
class OutputPlug(Plug):
	node : ThreadedDevice = None
	pass
# endregion


# define node class
class ThreadedDevice(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	frameRate_ : FrameRatePlug = PlugDescriptor("frameRate")
	live_ : LivePlug = PlugDescriptor("live")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "threadedDevice"
	apiTypeInt = 1076
	apiTypeStr = "kThreadedDevice"
	typeIdInt = 1952998518
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "frameRate", "live", "output"]
	nodeLeafPlugs = ["binMembership", "frameRate", "live", "output"]
	pass

