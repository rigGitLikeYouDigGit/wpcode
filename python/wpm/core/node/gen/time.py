

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
	node : Time = None
	pass
class EnableTimewarpPlug(Plug):
	node : Time = None
	pass
class OutTimePlug(Plug):
	node : Time = None
	pass
class TimecodeMayaStartPlug(Plug):
	node : Time = None
	pass
class TimecodeProductionStartPlug(Plug):
	node : Time = None
	pass
class TimewarpIn_HiddenPlug(Plug):
	parent : TimewarpInPlug = PlugDescriptor("timewarpIn")
	node : Time = None
	pass
class TimewarpIn_InmapFromPlug(Plug):
	parent : TimewarpIn_InmapPlug = PlugDescriptor("timewarpIn_Inmap")
	node : Time = None
	pass
class TimewarpIn_InmapToPlug(Plug):
	parent : TimewarpIn_InmapPlug = PlugDescriptor("timewarpIn_Inmap")
	node : Time = None
	pass
class TimewarpIn_InmapPlug(Plug):
	parent : TimewarpInPlug = PlugDescriptor("timewarpIn")
	timewarpIn_InmapFrom_ : TimewarpIn_InmapFromPlug = PlugDescriptor("timewarpIn_InmapFrom")
	twiif_ : TimewarpIn_InmapFromPlug = PlugDescriptor("timewarpIn_InmapFrom")
	timewarpIn_InmapTo_ : TimewarpIn_InmapToPlug = PlugDescriptor("timewarpIn_InmapTo")
	twiit_ : TimewarpIn_InmapToPlug = PlugDescriptor("timewarpIn_InmapTo")
	node : Time = None
	pass
class TimewarpIn_OutmapFromPlug(Plug):
	parent : TimewarpIn_OutmapPlug = PlugDescriptor("timewarpIn_Outmap")
	node : Time = None
	pass
class TimewarpIn_OutmapToPlug(Plug):
	parent : TimewarpIn_OutmapPlug = PlugDescriptor("timewarpIn_Outmap")
	node : Time = None
	pass
class TimewarpIn_OutmapPlug(Plug):
	parent : TimewarpInPlug = PlugDescriptor("timewarpIn")
	timewarpIn_OutmapFrom_ : TimewarpIn_OutmapFromPlug = PlugDescriptor("timewarpIn_OutmapFrom")
	twiof_ : TimewarpIn_OutmapFromPlug = PlugDescriptor("timewarpIn_OutmapFrom")
	timewarpIn_OutmapTo_ : TimewarpIn_OutmapToPlug = PlugDescriptor("timewarpIn_OutmapTo")
	twiot_ : TimewarpIn_OutmapToPlug = PlugDescriptor("timewarpIn_OutmapTo")
	node : Time = None
	pass
class TimewarpIn_RawPlug(Plug):
	parent : TimewarpInPlug = PlugDescriptor("timewarpIn")
	node : Time = None
	pass
class TimewarpInPlug(Plug):
	timewarpIn_Hidden_ : TimewarpIn_HiddenPlug = PlugDescriptor("timewarpIn_Hidden")
	twih_ : TimewarpIn_HiddenPlug = PlugDescriptor("timewarpIn_Hidden")
	timewarpIn_Inmap_ : TimewarpIn_InmapPlug = PlugDescriptor("timewarpIn_Inmap")
	twii_ : TimewarpIn_InmapPlug = PlugDescriptor("timewarpIn_Inmap")
	timewarpIn_Outmap_ : TimewarpIn_OutmapPlug = PlugDescriptor("timewarpIn_Outmap")
	twio_ : TimewarpIn_OutmapPlug = PlugDescriptor("timewarpIn_Outmap")
	timewarpIn_Raw_ : TimewarpIn_RawPlug = PlugDescriptor("timewarpIn_Raw")
	twir_ : TimewarpIn_RawPlug = PlugDescriptor("timewarpIn_Raw")
	node : Time = None
	pass
class UnwarpedTimePlug(Plug):
	node : Time = None
	pass
# endregion


# define node class
class Time(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	enableTimewarp_ : EnableTimewarpPlug = PlugDescriptor("enableTimewarp")
	outTime_ : OutTimePlug = PlugDescriptor("outTime")
	timecodeMayaStart_ : TimecodeMayaStartPlug = PlugDescriptor("timecodeMayaStart")
	timecodeProductionStart_ : TimecodeProductionStartPlug = PlugDescriptor("timecodeProductionStart")
	timewarpIn_Hidden_ : TimewarpIn_HiddenPlug = PlugDescriptor("timewarpIn_Hidden")
	timewarpIn_InmapFrom_ : TimewarpIn_InmapFromPlug = PlugDescriptor("timewarpIn_InmapFrom")
	timewarpIn_InmapTo_ : TimewarpIn_InmapToPlug = PlugDescriptor("timewarpIn_InmapTo")
	timewarpIn_Inmap_ : TimewarpIn_InmapPlug = PlugDescriptor("timewarpIn_Inmap")
	timewarpIn_OutmapFrom_ : TimewarpIn_OutmapFromPlug = PlugDescriptor("timewarpIn_OutmapFrom")
	timewarpIn_OutmapTo_ : TimewarpIn_OutmapToPlug = PlugDescriptor("timewarpIn_OutmapTo")
	timewarpIn_Outmap_ : TimewarpIn_OutmapPlug = PlugDescriptor("timewarpIn_Outmap")
	timewarpIn_Raw_ : TimewarpIn_RawPlug = PlugDescriptor("timewarpIn_Raw")
	timewarpIn_ : TimewarpInPlug = PlugDescriptor("timewarpIn")
	unwarpedTime_ : UnwarpedTimePlug = PlugDescriptor("unwarpedTime")

	# node attributes

	typeName = "time"
	apiTypeInt = 520
	apiTypeStr = "kTime"
	typeIdInt = 1414090053
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "enableTimewarp", "outTime", "timecodeMayaStart", "timecodeProductionStart", "timewarpIn_Hidden", "timewarpIn_InmapFrom", "timewarpIn_InmapTo", "timewarpIn_Inmap", "timewarpIn_OutmapFrom", "timewarpIn_OutmapTo", "timewarpIn_Outmap", "timewarpIn_Raw", "timewarpIn", "unwarpedTime"]
	nodeLeafPlugs = ["binMembership", "enableTimewarp", "outTime", "timecodeMayaStart", "timecodeProductionStart", "timewarpIn", "unwarpedTime"]
	pass

