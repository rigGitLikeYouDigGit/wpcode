

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
	node : TimeEditorTracks = None
	pass
class ClipPlug(Plug):
	node : TimeEditorTracks = None
	pass
class ClipTimePlug(Plug):
	node : TimeEditorTracks = None
	pass
class ClipchangedPlug(Plug):
	node : TimeEditorTracks = None
	pass
class CompositionPlug(Plug):
	node : TimeEditorTracks = None
	pass
class CrossfadeClipId1Plug(Plug):
	parent : CrossfadePlug = PlugDescriptor("crossfade")
	node : TimeEditorTracks = None
	pass
class CrossfadeClipId2Plug(Plug):
	parent : CrossfadePlug = PlugDescriptor("crossfade")
	node : TimeEditorTracks = None
	pass
class CrossfadeCurvePlug(Plug):
	parent : CrossfadePlug = PlugDescriptor("crossfade")
	node : TimeEditorTracks = None
	pass
class CrossfadeModePlug(Plug):
	parent : CrossfadePlug = PlugDescriptor("crossfade")
	node : TimeEditorTracks = None
	pass
class CrossfadePlug(Plug):
	crossfadeClipId1_ : CrossfadeClipId1Plug = PlugDescriptor("crossfadeClipId1")
	cid1_ : CrossfadeClipId1Plug = PlugDescriptor("crossfadeClipId1")
	crossfadeClipId2_ : CrossfadeClipId2Plug = PlugDescriptor("crossfadeClipId2")
	cid2_ : CrossfadeClipId2Plug = PlugDescriptor("crossfadeClipId2")
	crossfadeCurve_ : CrossfadeCurvePlug = PlugDescriptor("crossfadeCurve")
	cc_ : CrossfadeCurvePlug = PlugDescriptor("crossfadeCurve")
	crossfadeMode_ : CrossfadeModePlug = PlugDescriptor("crossfadeMode")
	cm_ : CrossfadeModePlug = PlugDescriptor("crossfadeMode")
	node : TimeEditorTracks = None
	pass
class MutedPlug(Plug):
	node : TimeEditorTracks = None
	pass
class ParentTimePlug(Plug):
	node : TimeEditorTracks = None
	pass
class StatePlug(Plug):
	node : TimeEditorTracks = None
	pass
class IndexPlug(Plug):
	parent : TrackPlug = PlugDescriptor("track")
	node : TimeEditorTracks = None
	pass
class TrackColorBPlug(Plug):
	parent : TrackColorPlug = PlugDescriptor("trackColor")
	node : TimeEditorTracks = None
	pass
class TrackColorGPlug(Plug):
	parent : TrackColorPlug = PlugDescriptor("trackColor")
	node : TimeEditorTracks = None
	pass
class TrackColorRPlug(Plug):
	parent : TrackColorPlug = PlugDescriptor("trackColor")
	node : TimeEditorTracks = None
	pass
class TrackColorPlug(Plug):
	parent : TrackPlug = PlugDescriptor("track")
	trackColorB_ : TrackColorBPlug = PlugDescriptor("trackColorB")
	tcb_ : TrackColorBPlug = PlugDescriptor("trackColorB")
	trackColorG_ : TrackColorGPlug = PlugDescriptor("trackColorG")
	tcg_ : TrackColorGPlug = PlugDescriptor("trackColorG")
	trackColorR_ : TrackColorRPlug = PlugDescriptor("trackColorR")
	tcr_ : TrackColorRPlug = PlugDescriptor("trackColorR")
	node : TimeEditorTracks = None
	pass
class TrackGhostPlug(Plug):
	parent : TrackPlug = PlugDescriptor("track")
	node : TimeEditorTracks = None
	pass
class TrackHeightPlug(Plug):
	parent : TrackPlug = PlugDescriptor("track")
	node : TimeEditorTracks = None
	pass
class TrackMutedPlug(Plug):
	parent : TrackPlug = PlugDescriptor("track")
	node : TimeEditorTracks = None
	pass
class TrackNamePlug(Plug):
	parent : TrackPlug = PlugDescriptor("track")
	node : TimeEditorTracks = None
	pass
class TrackSoloPlug(Plug):
	parent : TrackPlug = PlugDescriptor("track")
	node : TimeEditorTracks = None
	pass
class TrackSoloMutePlug(Plug):
	parent : TrackPlug = PlugDescriptor("track")
	node : TimeEditorTracks = None
	pass
class TypePlug(Plug):
	parent : TrackPlug = PlugDescriptor("track")
	node : TimeEditorTracks = None
	pass
class UseTrackColorPlug(Plug):
	parent : TrackPlug = PlugDescriptor("track")
	node : TimeEditorTracks = None
	pass
class TrackPlug(Plug):
	index_ : IndexPlug = PlugDescriptor("index")
	idx_ : IndexPlug = PlugDescriptor("index")
	trackColor_ : TrackColorPlug = PlugDescriptor("trackColor")
	tc_ : TrackColorPlug = PlugDescriptor("trackColor")
	trackGhost_ : TrackGhostPlug = PlugDescriptor("trackGhost")
	tgh_ : TrackGhostPlug = PlugDescriptor("trackGhost")
	trackHeight_ : TrackHeightPlug = PlugDescriptor("trackHeight")
	th_ : TrackHeightPlug = PlugDescriptor("trackHeight")
	trackMuted_ : TrackMutedPlug = PlugDescriptor("trackMuted")
	tm_ : TrackMutedPlug = PlugDescriptor("trackMuted")
	trackName_ : TrackNamePlug = PlugDescriptor("trackName")
	n_ : TrackNamePlug = PlugDescriptor("trackName")
	trackSolo_ : TrackSoloPlug = PlugDescriptor("trackSolo")
	ts_ : TrackSoloPlug = PlugDescriptor("trackSolo")
	trackSoloMute_ : TrackSoloMutePlug = PlugDescriptor("trackSoloMute")
	tsm_ : TrackSoloMutePlug = PlugDescriptor("trackSoloMute")
	type_ : TypePlug = PlugDescriptor("type")
	typ_ : TypePlug = PlugDescriptor("type")
	useTrackColor_ : UseTrackColorPlug = PlugDescriptor("useTrackColor")
	utc_ : UseTrackColorPlug = PlugDescriptor("useTrackColor")
	node : TimeEditorTracks = None
	pass
# endregion


# define node class
class TimeEditorTracks(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	clip_ : ClipPlug = PlugDescriptor("clip")
	clipTime_ : ClipTimePlug = PlugDescriptor("clipTime")
	clipchanged_ : ClipchangedPlug = PlugDescriptor("clipchanged")
	composition_ : CompositionPlug = PlugDescriptor("composition")
	crossfadeClipId1_ : CrossfadeClipId1Plug = PlugDescriptor("crossfadeClipId1")
	crossfadeClipId2_ : CrossfadeClipId2Plug = PlugDescriptor("crossfadeClipId2")
	crossfadeCurve_ : CrossfadeCurvePlug = PlugDescriptor("crossfadeCurve")
	crossfadeMode_ : CrossfadeModePlug = PlugDescriptor("crossfadeMode")
	crossfade_ : CrossfadePlug = PlugDescriptor("crossfade")
	muted_ : MutedPlug = PlugDescriptor("muted")
	parentTime_ : ParentTimePlug = PlugDescriptor("parentTime")
	state_ : StatePlug = PlugDescriptor("state")
	index_ : IndexPlug = PlugDescriptor("index")
	trackColorB_ : TrackColorBPlug = PlugDescriptor("trackColorB")
	trackColorG_ : TrackColorGPlug = PlugDescriptor("trackColorG")
	trackColorR_ : TrackColorRPlug = PlugDescriptor("trackColorR")
	trackColor_ : TrackColorPlug = PlugDescriptor("trackColor")
	trackGhost_ : TrackGhostPlug = PlugDescriptor("trackGhost")
	trackHeight_ : TrackHeightPlug = PlugDescriptor("trackHeight")
	trackMuted_ : TrackMutedPlug = PlugDescriptor("trackMuted")
	trackName_ : TrackNamePlug = PlugDescriptor("trackName")
	trackSolo_ : TrackSoloPlug = PlugDescriptor("trackSolo")
	trackSoloMute_ : TrackSoloMutePlug = PlugDescriptor("trackSoloMute")
	type_ : TypePlug = PlugDescriptor("type")
	useTrackColor_ : UseTrackColorPlug = PlugDescriptor("useTrackColor")
	track_ : TrackPlug = PlugDescriptor("track")

	# node attributes

	typeName = "timeEditorTracks"
	apiTypeInt = 1107
	apiTypeStr = "kTimeEditorTracks"
	typeIdInt = 1413829707
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "clip", "clipTime", "clipchanged", "composition", "crossfadeClipId1", "crossfadeClipId2", "crossfadeCurve", "crossfadeMode", "crossfade", "muted", "parentTime", "state", "index", "trackColorB", "trackColorG", "trackColorR", "trackColor", "trackGhost", "trackHeight", "trackMuted", "trackName", "trackSolo", "trackSoloMute", "type", "useTrackColor", "track"]
	nodeLeafPlugs = ["binMembership", "clip", "clipTime", "clipchanged", "composition", "crossfade", "muted", "parentTime", "state", "track"]
	pass

