

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : Audio = None
	pass
class ChannelsPlug(Plug):
	node : Audio = None
	pass
class DurationPlug(Plug):
	node : Audio = None
	pass
class EndFramePlug(Plug):
	node : Audio = None
	pass
class FilenamePlug(Plug):
	node : Audio = None
	pass
class FrameCountPlug(Plug):
	node : Audio = None
	pass
class MutePlug(Plug):
	node : Audio = None
	pass
class OffsetPlug(Plug):
	node : Audio = None
	pass
class OrderPlug(Plug):
	node : Audio = None
	pass
class SampleRatePlug(Plug):
	node : Audio = None
	pass
class SilencePlug(Plug):
	node : Audio = None
	pass
class SourceEndPlug(Plug):
	node : Audio = None
	pass
class SourceStartPlug(Plug):
	node : Audio = None
	pass
class TrackPlug(Plug):
	node : Audio = None
	pass
class TrackStatePlug(Plug):
	node : Audio = None
	pass
# endregion


# define node class
class Audio(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	channels_ : ChannelsPlug = PlugDescriptor("channels")
	duration_ : DurationPlug = PlugDescriptor("duration")
	endFrame_ : EndFramePlug = PlugDescriptor("endFrame")
	filename_ : FilenamePlug = PlugDescriptor("filename")
	frameCount_ : FrameCountPlug = PlugDescriptor("frameCount")
	mute_ : MutePlug = PlugDescriptor("mute")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	order_ : OrderPlug = PlugDescriptor("order")
	sampleRate_ : SampleRatePlug = PlugDescriptor("sampleRate")
	silence_ : SilencePlug = PlugDescriptor("silence")
	sourceEnd_ : SourceEndPlug = PlugDescriptor("sourceEnd")
	sourceStart_ : SourceStartPlug = PlugDescriptor("sourceStart")
	track_ : TrackPlug = PlugDescriptor("track")
	trackState_ : TrackStatePlug = PlugDescriptor("trackState")

	# node attributes

	typeName = "audio"
	apiTypeInt = 22
	apiTypeStr = "kAudio"
	typeIdInt = 1096107081
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "channels", "duration", "endFrame", "filename", "frameCount", "mute", "offset", "order", "sampleRate", "silence", "sourceEnd", "sourceStart", "track", "trackState"]
	nodeLeafPlugs = ["binMembership", "channels", "duration", "endFrame", "filename", "frameCount", "mute", "offset", "order", "sampleRate", "silence", "sourceEnd", "sourceStart", "track", "trackState"]
	pass

