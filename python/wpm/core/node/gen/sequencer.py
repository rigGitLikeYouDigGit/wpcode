

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
class AudioPlug(Plug):
	node : Sequencer = None
	pass
class BinMembershipPlug(Plug):
	node : Sequencer = None
	pass
class MaxFramePlug(Plug):
	node : Sequencer = None
	pass
class MinFramePlug(Plug):
	node : Sequencer = None
	pass
class ShotsPlug(Plug):
	node : Sequencer = None
	pass
# endregion


# define node class
class Sequencer(_BASE_):
	audio_ : AudioPlug = PlugDescriptor("audio")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	maxFrame_ : MaxFramePlug = PlugDescriptor("maxFrame")
	minFrame_ : MinFramePlug = PlugDescriptor("minFrame")
	shots_ : ShotsPlug = PlugDescriptor("shots")

	# node attributes

	typeName = "sequencer"
	apiTypeInt = 1050
	apiTypeStr = "kSequencer"
	typeIdInt = 1397837379
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["audio", "binMembership", "maxFrame", "minFrame", "shots"]
	nodeLeafPlugs = ["audio", "binMembership", "maxFrame", "minFrame", "shots"]
	pass

