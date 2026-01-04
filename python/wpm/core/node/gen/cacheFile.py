

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	CacheBase = Catalogue.CacheBase
else:
	from .. import retriever
	CacheBase = retriever.getNodeCls("CacheBase")
	assert CacheBase

# add node doc



# region plug type defs
class CacheNamePlug(Plug):
	node : CacheFile = None
	pass
class CachePathPlug(Plug):
	node : CacheFile = None
	pass
class CacheWeightsPlug(Plug):
	node : CacheFile = None
	pass
class ChannelPlug(Plug):
	node : CacheFile = None
	pass
class DisplayLoadProgressPlug(Plug):
	node : CacheFile = None
	pass
class EnablePlug(Plug):
	node : CacheFile = None
	pass
class EndPlug(Plug):
	node : CacheFile = None
	pass
class FormatPlug(Plug):
	node : CacheFile = None
	pass
class HoldPlug(Plug):
	node : CacheFile = None
	pass
class MemQueueSizePlug(Plug):
	node : CacheFile = None
	pass
class MultiThreadPlug(Plug):
	node : CacheFile = None
	pass
class OriginalEndPlug(Plug):
	node : CacheFile = None
	pass
class OriginalStartPlug(Plug):
	node : CacheFile = None
	pass
class OscillatePlug(Plug):
	node : CacheFile = None
	pass
class PerPtWeightsPlug(Plug):
	node : CacheFile = None
	pass
class PostCyclePlug(Plug):
	node : CacheFile = None
	pass
class PreCyclePlug(Plug):
	node : CacheFile = None
	pass
class ReversePlug(Plug):
	node : CacheFile = None
	pass
class ScalePlug(Plug):
	node : CacheFile = None
	pass
class SourceEndPlug(Plug):
	node : CacheFile = None
	pass
class SourceStartPlug(Plug):
	node : CacheFile = None
	pass
class StartPlug(Plug):
	node : CacheFile = None
	pass
class StartFramePlug(Plug):
	node : CacheFile = None
	pass
class TimePlug(Plug):
	node : CacheFile = None
	pass
class TrackPlug(Plug):
	node : CacheFile = None
	pass
class TrackStatePlug(Plug):
	node : CacheFile = None
	pass
# endregion


# define node class
class CacheFile(CacheBase):
	cacheName_ : CacheNamePlug = PlugDescriptor("cacheName")
	cachePath_ : CachePathPlug = PlugDescriptor("cachePath")
	cacheWeights_ : CacheWeightsPlug = PlugDescriptor("cacheWeights")
	channel_ : ChannelPlug = PlugDescriptor("channel")
	displayLoadProgress_ : DisplayLoadProgressPlug = PlugDescriptor("displayLoadProgress")
	enable_ : EnablePlug = PlugDescriptor("enable")
	end_ : EndPlug = PlugDescriptor("end")
	format_ : FormatPlug = PlugDescriptor("format")
	hold_ : HoldPlug = PlugDescriptor("hold")
	memQueueSize_ : MemQueueSizePlug = PlugDescriptor("memQueueSize")
	multiThread_ : MultiThreadPlug = PlugDescriptor("multiThread")
	originalEnd_ : OriginalEndPlug = PlugDescriptor("originalEnd")
	originalStart_ : OriginalStartPlug = PlugDescriptor("originalStart")
	oscillate_ : OscillatePlug = PlugDescriptor("oscillate")
	perPtWeights_ : PerPtWeightsPlug = PlugDescriptor("perPtWeights")
	postCycle_ : PostCyclePlug = PlugDescriptor("postCycle")
	preCycle_ : PreCyclePlug = PlugDescriptor("preCycle")
	reverse_ : ReversePlug = PlugDescriptor("reverse")
	scale_ : ScalePlug = PlugDescriptor("scale")
	sourceEnd_ : SourceEndPlug = PlugDescriptor("sourceEnd")
	sourceStart_ : SourceStartPlug = PlugDescriptor("sourceStart")
	start_ : StartPlug = PlugDescriptor("start")
	startFrame_ : StartFramePlug = PlugDescriptor("startFrame")
	time_ : TimePlug = PlugDescriptor("time")
	track_ : TrackPlug = PlugDescriptor("track")
	trackState_ : TrackStatePlug = PlugDescriptor("trackState")

	# node attributes

	typeName = "cacheFile"
	apiTypeInt = 987
	apiTypeStr = "kCacheFile"
	typeIdInt = 1128482886
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["cacheName", "cachePath", "cacheWeights", "channel", "displayLoadProgress", "enable", "end", "format", "hold", "memQueueSize", "multiThread", "originalEnd", "originalStart", "oscillate", "perPtWeights", "postCycle", "preCycle", "reverse", "scale", "sourceEnd", "sourceStart", "start", "startFrame", "time", "track", "trackState"]
	nodeLeafPlugs = ["cacheName", "cachePath", "cacheWeights", "channel", "displayLoadProgress", "enable", "end", "format", "hold", "memQueueSize", "multiThread", "originalEnd", "originalStart", "oscillate", "perPtWeights", "postCycle", "preCycle", "reverse", "scale", "sourceEnd", "sourceStart", "start", "startFrame", "time", "track", "trackState"]
	pass

