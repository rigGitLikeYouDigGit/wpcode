

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
	node : SequenceManager = None
	pass
class EnabledPlug(Plug):
	node : SequenceManager = None
	pass
class OutTimePlug(Plug):
	node : SequenceManager = None
	pass
class RangeEnabledPlug(Plug):
	node : SequenceManager = None
	pass
class RangeMaxPlug(Plug):
	node : SequenceManager = None
	pass
class RangeMinPlug(Plug):
	node : SequenceManager = None
	pass
class SequencesPlug(Plug):
	node : SequenceManager = None
	pass
class SkipGapsPlug(Plug):
	node : SequenceManager = None
	pass
class TrackInfoManagerPlug(Plug):
	node : SequenceManager = None
	pass
# endregion


# define node class
class SequenceManager(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	enabled_ : EnabledPlug = PlugDescriptor("enabled")
	outTime_ : OutTimePlug = PlugDescriptor("outTime")
	rangeEnabled_ : RangeEnabledPlug = PlugDescriptor("rangeEnabled")
	rangeMax_ : RangeMaxPlug = PlugDescriptor("rangeMax")
	rangeMin_ : RangeMinPlug = PlugDescriptor("rangeMin")
	sequences_ : SequencesPlug = PlugDescriptor("sequences")
	skipGaps_ : SkipGapsPlug = PlugDescriptor("skipGaps")
	trackInfoManager_ : TrackInfoManagerPlug = PlugDescriptor("trackInfoManager")

	# node attributes

	typeName = "sequenceManager"
	apiTypeInt = 1049
	apiTypeStr = "kSequenceManager"
	typeIdInt = 1397837127
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "enabled", "outTime", "rangeEnabled", "rangeMax", "rangeMin", "sequences", "skipGaps", "trackInfoManager"]
	nodeLeafPlugs = ["binMembership", "enabled", "outTime", "rangeEnabled", "rangeMax", "rangeMin", "sequences", "skipGaps", "trackInfoManager"]
	pass

