

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
	node : FrameCache = None
	pass
class FuturePlug(Plug):
	node : FrameCache = None
	pass
class PastPlug(Plug):
	node : FrameCache = None
	pass
class StreamPlug(Plug):
	node : FrameCache = None
	pass
class VaryTimePlug(Plug):
	node : FrameCache = None
	pass
class VaryingPlug(Plug):
	node : FrameCache = None
	pass
# endregion


# define node class
class FrameCache(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	future_ : FuturePlug = PlugDescriptor("future")
	past_ : PastPlug = PlugDescriptor("past")
	stream_ : StreamPlug = PlugDescriptor("stream")
	varyTime_ : VaryTimePlug = PlugDescriptor("varyTime")
	varying_ : VaryingPlug = PlugDescriptor("varying")

	# node attributes

	typeName = "frameCache"
	typeIdInt = 1178813256
	nodeLeafClassAttrs = ["binMembership", "future", "past", "stream", "varyTime", "varying"]
	nodeLeafPlugs = ["binMembership", "future", "past", "stream", "varyTime", "varying"]
	pass

