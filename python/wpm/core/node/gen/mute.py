

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
	node : Mute = None
	pass
class HoldPlug(Plug):
	node : Mute = None
	pass
class HoldTimePlug(Plug):
	node : Mute = None
	pass
class InputPlug(Plug):
	node : Mute = None
	pass
class MutePlug(Plug):
	node : Mute = None
	pass
class OutputPlug(Plug):
	node : Mute = None
	pass
# endregion


# define node class
class Mute(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	hold_ : HoldPlug = PlugDescriptor("hold")
	holdTime_ : HoldTimePlug = PlugDescriptor("holdTime")
	input_ : InputPlug = PlugDescriptor("input")
	mute_ : MutePlug = PlugDescriptor("mute")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "mute"
	apiTypeInt = 931
	apiTypeStr = "kMute"
	typeIdInt = 1297437765
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "hold", "holdTime", "input", "mute", "output"]
	nodeLeafPlugs = ["binMembership", "hold", "holdTime", "input", "mute", "output"]
	pass

