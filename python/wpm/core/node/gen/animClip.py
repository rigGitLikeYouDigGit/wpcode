

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
class AbsoluteChannelPlug(Plug):
	node : AnimClip = None
	pass
class AbsoluteRotationsPlug(Plug):
	node : AnimClip = None
	pass
class BinMembershipPlug(Plug):
	node : AnimClip = None
	pass
class ChannelOffsetPlug(Plug):
	node : AnimClip = None
	pass
class ClipPlug(Plug):
	node : AnimClip = None
	pass
class ClipDataPlug(Plug):
	node : AnimClip = None
	pass
class ClipInstancePlug(Plug):
	node : AnimClip = None
	pass
class CyclePlug(Plug):
	node : AnimClip = None
	pass
class DurationPlug(Plug):
	node : AnimClip = None
	pass
class EnablePlug(Plug):
	node : AnimClip = None
	pass
class HoldPlug(Plug):
	node : AnimClip = None
	pass
class LocalStartPositionXPlug(Plug):
	parent : LocalStartPositionPlug = PlugDescriptor("localStartPosition")
	node : AnimClip = None
	pass
class LocalStartPositionYPlug(Plug):
	parent : LocalStartPositionPlug = PlugDescriptor("localStartPosition")
	node : AnimClip = None
	pass
class LocalStartPositionZPlug(Plug):
	parent : LocalStartPositionPlug = PlugDescriptor("localStartPosition")
	node : AnimClip = None
	pass
class LocalStartPositionPlug(Plug):
	localStartPositionX_ : LocalStartPositionXPlug = PlugDescriptor("localStartPositionX")
	lspx_ : LocalStartPositionXPlug = PlugDescriptor("localStartPositionX")
	localStartPositionY_ : LocalStartPositionYPlug = PlugDescriptor("localStartPositionY")
	lspy_ : LocalStartPositionYPlug = PlugDescriptor("localStartPositionY")
	localStartPositionZ_ : LocalStartPositionZPlug = PlugDescriptor("localStartPositionZ")
	lspz_ : LocalStartPositionZPlug = PlugDescriptor("localStartPositionZ")
	node : AnimClip = None
	pass
class OffsetPlug(Plug):
	node : AnimClip = None
	pass
class OffsetXformPlug(Plug):
	node : AnimClip = None
	pass
class PosePlug(Plug):
	node : AnimClip = None
	pass
class PostCyclePlug(Plug):
	node : AnimClip = None
	pass
class PreCyclePlug(Plug):
	node : AnimClip = None
	pass
class RecomputeOffsetPlug(Plug):
	node : AnimClip = None
	pass
class ScalePlug(Plug):
	node : AnimClip = None
	pass
class SourceEndPlug(Plug):
	node : AnimClip = None
	pass
class SourceStartPlug(Plug):
	node : AnimClip = None
	pass
class StartPlug(Plug):
	node : AnimClip = None
	pass
class StartFramePlug(Plug):
	node : AnimClip = None
	pass
class StartPercentPlug(Plug):
	node : AnimClip = None
	pass
class StartTrimPlug(Plug):
	node : AnimClip = None
	pass
class TimeWarpPlug(Plug):
	node : AnimClip = None
	pass
class TimeWarpEnablePlug(Plug):
	node : AnimClip = None
	pass
class UseChannelOffsetPlug(Plug):
	node : AnimClip = None
	pass
class WeightPlug(Plug):
	node : AnimClip = None
	pass
class WeightStylePlug(Plug):
	node : AnimClip = None
	pass
class WorldStartPositionXPlug(Plug):
	parent : WorldStartPositionPlug = PlugDescriptor("worldStartPosition")
	node : AnimClip = None
	pass
class WorldStartPositionYPlug(Plug):
	parent : WorldStartPositionPlug = PlugDescriptor("worldStartPosition")
	node : AnimClip = None
	pass
class WorldStartPositionZPlug(Plug):
	parent : WorldStartPositionPlug = PlugDescriptor("worldStartPosition")
	node : AnimClip = None
	pass
class WorldStartPositionPlug(Plug):
	worldStartPositionX_ : WorldStartPositionXPlug = PlugDescriptor("worldStartPositionX")
	wspx_ : WorldStartPositionXPlug = PlugDescriptor("worldStartPositionX")
	worldStartPositionY_ : WorldStartPositionYPlug = PlugDescriptor("worldStartPositionY")
	wspy_ : WorldStartPositionYPlug = PlugDescriptor("worldStartPositionY")
	worldStartPositionZ_ : WorldStartPositionZPlug = PlugDescriptor("worldStartPositionZ")
	wspz_ : WorldStartPositionZPlug = PlugDescriptor("worldStartPositionZ")
	node : AnimClip = None
	pass
# endregion


# define node class
class AnimClip(_BASE_):
	absoluteChannel_ : AbsoluteChannelPlug = PlugDescriptor("absoluteChannel")
	absoluteRotations_ : AbsoluteRotationsPlug = PlugDescriptor("absoluteRotations")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	channelOffset_ : ChannelOffsetPlug = PlugDescriptor("channelOffset")
	clip_ : ClipPlug = PlugDescriptor("clip")
	clipData_ : ClipDataPlug = PlugDescriptor("clipData")
	clipInstance_ : ClipInstancePlug = PlugDescriptor("clipInstance")
	cycle_ : CyclePlug = PlugDescriptor("cycle")
	duration_ : DurationPlug = PlugDescriptor("duration")
	enable_ : EnablePlug = PlugDescriptor("enable")
	hold_ : HoldPlug = PlugDescriptor("hold")
	localStartPositionX_ : LocalStartPositionXPlug = PlugDescriptor("localStartPositionX")
	localStartPositionY_ : LocalStartPositionYPlug = PlugDescriptor("localStartPositionY")
	localStartPositionZ_ : LocalStartPositionZPlug = PlugDescriptor("localStartPositionZ")
	localStartPosition_ : LocalStartPositionPlug = PlugDescriptor("localStartPosition")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	offsetXform_ : OffsetXformPlug = PlugDescriptor("offsetXform")
	pose_ : PosePlug = PlugDescriptor("pose")
	postCycle_ : PostCyclePlug = PlugDescriptor("postCycle")
	preCycle_ : PreCyclePlug = PlugDescriptor("preCycle")
	recomputeOffset_ : RecomputeOffsetPlug = PlugDescriptor("recomputeOffset")
	scale_ : ScalePlug = PlugDescriptor("scale")
	sourceEnd_ : SourceEndPlug = PlugDescriptor("sourceEnd")
	sourceStart_ : SourceStartPlug = PlugDescriptor("sourceStart")
	start_ : StartPlug = PlugDescriptor("start")
	startFrame_ : StartFramePlug = PlugDescriptor("startFrame")
	startPercent_ : StartPercentPlug = PlugDescriptor("startPercent")
	startTrim_ : StartTrimPlug = PlugDescriptor("startTrim")
	timeWarp_ : TimeWarpPlug = PlugDescriptor("timeWarp")
	timeWarpEnable_ : TimeWarpEnablePlug = PlugDescriptor("timeWarpEnable")
	useChannelOffset_ : UseChannelOffsetPlug = PlugDescriptor("useChannelOffset")
	weight_ : WeightPlug = PlugDescriptor("weight")
	weightStyle_ : WeightStylePlug = PlugDescriptor("weightStyle")
	worldStartPositionX_ : WorldStartPositionXPlug = PlugDescriptor("worldStartPositionX")
	worldStartPositionY_ : WorldStartPositionYPlug = PlugDescriptor("worldStartPositionY")
	worldStartPositionZ_ : WorldStartPositionZPlug = PlugDescriptor("worldStartPositionZ")
	worldStartPosition_ : WorldStartPositionPlug = PlugDescriptor("worldStartPosition")

	# node attributes

	typeName = "animClip"
	typeIdInt = 1129074766
	nodeLeafClassAttrs = ["absoluteChannel", "absoluteRotations", "binMembership", "channelOffset", "clip", "clipData", "clipInstance", "cycle", "duration", "enable", "hold", "localStartPositionX", "localStartPositionY", "localStartPositionZ", "localStartPosition", "offset", "offsetXform", "pose", "postCycle", "preCycle", "recomputeOffset", "scale", "sourceEnd", "sourceStart", "start", "startFrame", "startPercent", "startTrim", "timeWarp", "timeWarpEnable", "useChannelOffset", "weight", "weightStyle", "worldStartPositionX", "worldStartPositionY", "worldStartPositionZ", "worldStartPosition"]
	nodeLeafPlugs = ["absoluteChannel", "absoluteRotations", "binMembership", "channelOffset", "clip", "clipData", "clipInstance", "cycle", "duration", "enable", "hold", "localStartPosition", "offset", "offsetXform", "pose", "postCycle", "preCycle", "recomputeOffset", "scale", "sourceEnd", "sourceStart", "start", "startFrame", "startPercent", "startTrim", "timeWarp", "timeWarpEnable", "useChannelOffset", "weight", "weightStyle", "worldStartPosition"]
	pass

