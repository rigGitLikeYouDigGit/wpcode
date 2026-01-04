

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
	node : DiskCache = None
	pass
class CacheNamePlug(Plug):
	node : DiskCache = None
	pass
class CacheTypePlug(Plug):
	node : DiskCache = None
	pass
class CopyLocallyPlug(Plug):
	node : DiskCache = None
	pass
class DiskCachePlug(Plug):
	node : DiskCache = None
	pass
class EnablePlug(Plug):
	node : DiskCache = None
	pass
class EndTimePlug(Plug):
	node : DiskCache = None
	pass
class HiddenCacheNamePlug(Plug):
	node : DiskCache = None
	pass
class SamplingRatePlug(Plug):
	node : DiskCache = None
	pass
class SamplingTypePlug(Plug):
	node : DiskCache = None
	pass
class StartTimePlug(Plug):
	node : DiskCache = None
	pass
# endregion


# define node class
class DiskCache(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	cacheName_ : CacheNamePlug = PlugDescriptor("cacheName")
	cacheType_ : CacheTypePlug = PlugDescriptor("cacheType")
	copyLocally_ : CopyLocallyPlug = PlugDescriptor("copyLocally")
	diskCache_ : DiskCachePlug = PlugDescriptor("diskCache")
	enable_ : EnablePlug = PlugDescriptor("enable")
	endTime_ : EndTimePlug = PlugDescriptor("endTime")
	hiddenCacheName_ : HiddenCacheNamePlug = PlugDescriptor("hiddenCacheName")
	samplingRate_ : SamplingRatePlug = PlugDescriptor("samplingRate")
	samplingType_ : SamplingTypePlug = PlugDescriptor("samplingType")
	startTime_ : StartTimePlug = PlugDescriptor("startTime")

	# node attributes

	typeName = "diskCache"
	apiTypeInt = 863
	apiTypeStr = "kDiskCache"
	typeIdInt = 1146309443
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "cacheName", "cacheType", "copyLocally", "diskCache", "enable", "endTime", "hiddenCacheName", "samplingRate", "samplingType", "startTime"]
	nodeLeafPlugs = ["binMembership", "cacheName", "cacheType", "copyLocally", "diskCache", "enable", "endTime", "hiddenCacheName", "samplingRate", "samplingType", "startTime"]
	pass

