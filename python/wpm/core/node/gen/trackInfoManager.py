

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
class AudioTitlePlug(Plug):
	parent : AudioTrackInfoPlug = PlugDescriptor("audioTrackInfo")
	node : TrackInfoManager = None
	pass
class AudioTrackInfoPlug(Plug):
	audioTitle_ : AudioTitlePlug = PlugDescriptor("audioTitle")
	at_ : AudioTitlePlug = PlugDescriptor("audioTitle")
	node : TrackInfoManager = None
	pass
class BinMembershipPlug(Plug):
	node : TrackInfoManager = None
	pass
class TitlePlug(Plug):
	parent : TrackInfoPlug = PlugDescriptor("trackInfo")
	node : TrackInfoManager = None
	pass
class TrackInfoPlug(Plug):
	title_ : TitlePlug = PlugDescriptor("title")
	t_ : TitlePlug = PlugDescriptor("title")
	node : TrackInfoManager = None
	pass
# endregion


# define node class
class TrackInfoManager(_BASE_):
	audioTitle_ : AudioTitlePlug = PlugDescriptor("audioTitle")
	audioTrackInfo_ : AudioTrackInfoPlug = PlugDescriptor("audioTrackInfo")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	title_ : TitlePlug = PlugDescriptor("title")
	trackInfo_ : TrackInfoPlug = PlugDescriptor("trackInfo")

	# node attributes

	typeName = "trackInfoManager"
	apiTypeInt = 1123
	apiTypeStr = "kTrackInfoManager"
	typeIdInt = 1414090055
	MFnCls = om.MFnDependencyNode
	pass

