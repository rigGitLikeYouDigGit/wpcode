

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
class AbsolutePlug(Plug):
	node : ClipScheduler = None
	pass
class AbsoluteRotationsPlug(Plug):
	node : ClipScheduler = None
	pass
class BinMembershipPlug(Plug):
	node : ClipScheduler = None
	pass
class FirstClipPlug(Plug):
	parent : BlendClipsPlug = PlugDescriptor("blendClips")
	node : ClipScheduler = None
	pass
class SecondClipPlug(Plug):
	parent : BlendClipsPlug = PlugDescriptor("blendClips")
	node : ClipScheduler = None
	pass
class BlendClipsPlug(Plug):
	firstClip_ : FirstClipPlug = PlugDescriptor("firstClip")
	fcl_ : FirstClipPlug = PlugDescriptor("firstClip")
	secondClip_ : SecondClipPlug = PlugDescriptor("secondClip")
	scl_ : SecondClipPlug = PlugDescriptor("secondClip")
	node : ClipScheduler = None
	pass
class BlendList_HiddenPlug(Plug):
	parent : BlendListPlug = PlugDescriptor("blendList")
	node : ClipScheduler = None
	pass
class BlendList_InmapFromPlug(Plug):
	parent : BlendList_InmapPlug = PlugDescriptor("blendList_Inmap")
	node : ClipScheduler = None
	pass
class BlendList_InmapToPlug(Plug):
	parent : BlendList_InmapPlug = PlugDescriptor("blendList_Inmap")
	node : ClipScheduler = None
	pass
class BlendList_InmapPlug(Plug):
	parent : BlendListPlug = PlugDescriptor("blendList")
	blendList_InmapFrom_ : BlendList_InmapFromPlug = PlugDescriptor("blendList_InmapFrom")
	blif_ : BlendList_InmapFromPlug = PlugDescriptor("blendList_InmapFrom")
	blendList_InmapTo_ : BlendList_InmapToPlug = PlugDescriptor("blendList_InmapTo")
	blit_ : BlendList_InmapToPlug = PlugDescriptor("blendList_InmapTo")
	node : ClipScheduler = None
	pass
class BlendList_OutmapFromPlug(Plug):
	parent : BlendList_OutmapPlug = PlugDescriptor("blendList_Outmap")
	node : ClipScheduler = None
	pass
class BlendList_OutmapToPlug(Plug):
	parent : BlendList_OutmapPlug = PlugDescriptor("blendList_Outmap")
	node : ClipScheduler = None
	pass
class BlendList_OutmapPlug(Plug):
	parent : BlendListPlug = PlugDescriptor("blendList")
	blendList_OutmapFrom_ : BlendList_OutmapFromPlug = PlugDescriptor("blendList_OutmapFrom")
	blof_ : BlendList_OutmapFromPlug = PlugDescriptor("blendList_OutmapFrom")
	blendList_OutmapTo_ : BlendList_OutmapToPlug = PlugDescriptor("blendList_OutmapTo")
	blot_ : BlendList_OutmapToPlug = PlugDescriptor("blendList_OutmapTo")
	node : ClipScheduler = None
	pass
class BlendList_RawPlug(Plug):
	parent : BlendListPlug = PlugDescriptor("blendList")
	node : ClipScheduler = None
	pass
class BlendListPlug(Plug):
	blendList_Hidden_ : BlendList_HiddenPlug = PlugDescriptor("blendList_Hidden")
	blh_ : BlendList_HiddenPlug = PlugDescriptor("blendList_Hidden")
	blendList_Inmap_ : BlendList_InmapPlug = PlugDescriptor("blendList_Inmap")
	bli_ : BlendList_InmapPlug = PlugDescriptor("blendList_Inmap")
	blendList_Outmap_ : BlendList_OutmapPlug = PlugDescriptor("blendList_Outmap")
	blo_ : BlendList_OutmapPlug = PlugDescriptor("blendList_Outmap")
	blendList_Raw_ : BlendList_RawPlug = PlugDescriptor("blendList_Raw")
	blr_ : BlendList_RawPlug = PlugDescriptor("blendList_Raw")
	node : ClipScheduler = None
	pass
class ClipPlug(Plug):
	node : ClipScheduler = None
	pass
class ClipEvaluatePlug(Plug):
	node : ClipScheduler = None
	pass
class ClipFunction_HiddenPlug(Plug):
	parent : ClipFunctionPlug = PlugDescriptor("clipFunction")
	node : ClipScheduler = None
	pass
class ClipFunction_InmapFromPlug(Plug):
	parent : ClipFunction_InmapPlug = PlugDescriptor("clipFunction_Inmap")
	node : ClipScheduler = None
	pass
class ClipFunction_InmapToPlug(Plug):
	parent : ClipFunction_InmapPlug = PlugDescriptor("clipFunction_Inmap")
	node : ClipScheduler = None
	pass
class ClipFunction_InmapPlug(Plug):
	parent : ClipFunctionPlug = PlugDescriptor("clipFunction")
	clipFunction_InmapFrom_ : ClipFunction_InmapFromPlug = PlugDescriptor("clipFunction_InmapFrom")
	cfif_ : ClipFunction_InmapFromPlug = PlugDescriptor("clipFunction_InmapFrom")
	clipFunction_InmapTo_ : ClipFunction_InmapToPlug = PlugDescriptor("clipFunction_InmapTo")
	cfit_ : ClipFunction_InmapToPlug = PlugDescriptor("clipFunction_InmapTo")
	node : ClipScheduler = None
	pass
class ClipFunction_OutmapFromPlug(Plug):
	parent : ClipFunction_OutmapPlug = PlugDescriptor("clipFunction_Outmap")
	node : ClipScheduler = None
	pass
class ClipFunction_OutmapToPlug(Plug):
	parent : ClipFunction_OutmapPlug = PlugDescriptor("clipFunction_Outmap")
	node : ClipScheduler = None
	pass
class ClipFunction_OutmapPlug(Plug):
	parent : ClipFunctionPlug = PlugDescriptor("clipFunction")
	clipFunction_OutmapFrom_ : ClipFunction_OutmapFromPlug = PlugDescriptor("clipFunction_OutmapFrom")
	cfof_ : ClipFunction_OutmapFromPlug = PlugDescriptor("clipFunction_OutmapFrom")
	clipFunction_OutmapTo_ : ClipFunction_OutmapToPlug = PlugDescriptor("clipFunction_OutmapTo")
	cfot_ : ClipFunction_OutmapToPlug = PlugDescriptor("clipFunction_OutmapTo")
	node : ClipScheduler = None
	pass
class ClipFunction_RawPlug(Plug):
	parent : ClipFunctionPlug = PlugDescriptor("clipFunction")
	node : ClipScheduler = None
	pass
class ClipFunctionPlug(Plug):
	clipFunction_Hidden_ : ClipFunction_HiddenPlug = PlugDescriptor("clipFunction_Hidden")
	cfh_ : ClipFunction_HiddenPlug = PlugDescriptor("clipFunction_Hidden")
	clipFunction_Inmap_ : ClipFunction_InmapPlug = PlugDescriptor("clipFunction_Inmap")
	cfi_ : ClipFunction_InmapPlug = PlugDescriptor("clipFunction_Inmap")
	clipFunction_Outmap_ : ClipFunction_OutmapPlug = PlugDescriptor("clipFunction_Outmap")
	cfo_ : ClipFunction_OutmapPlug = PlugDescriptor("clipFunction_Outmap")
	clipFunction_Raw_ : ClipFunction_RawPlug = PlugDescriptor("clipFunction_Raw")
	cfr_ : ClipFunction_RawPlug = PlugDescriptor("clipFunction_Raw")
	node : ClipScheduler = None
	pass
class ClipStatePercentEvalPlug(Plug):
	node : ClipScheduler = None
	pass
class CyclePlug(Plug):
	node : ClipScheduler = None
	pass
class EnablePlug(Plug):
	node : ClipScheduler = None
	pass
class HoldPlug(Plug):
	node : ClipScheduler = None
	pass
class NumTracksPlug(Plug):
	node : ClipScheduler = None
	pass
class PostCyclePlug(Plug):
	node : ClipScheduler = None
	pass
class PreCyclePlug(Plug):
	node : ClipScheduler = None
	pass
class ScalePlug(Plug):
	node : ClipScheduler = None
	pass
class SourceEndPlug(Plug):
	node : ClipScheduler = None
	pass
class SourceStartPlug(Plug):
	node : ClipScheduler = None
	pass
class StartPlug(Plug):
	node : ClipScheduler = None
	pass
class StartPercentPlug(Plug):
	node : ClipScheduler = None
	pass
class TrackPlug(Plug):
	node : ClipScheduler = None
	pass
class TrackStatePlug(Plug):
	node : ClipScheduler = None
	pass
class WeightPlug(Plug):
	node : ClipScheduler = None
	pass
class WeightStylePlug(Plug):
	node : ClipScheduler = None
	pass
# endregion


# define node class
class ClipScheduler(_BASE_):
	absolute_ : AbsolutePlug = PlugDescriptor("absolute")
	absoluteRotations_ : AbsoluteRotationsPlug = PlugDescriptor("absoluteRotations")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	firstClip_ : FirstClipPlug = PlugDescriptor("firstClip")
	secondClip_ : SecondClipPlug = PlugDescriptor("secondClip")
	blendClips_ : BlendClipsPlug = PlugDescriptor("blendClips")
	blendList_Hidden_ : BlendList_HiddenPlug = PlugDescriptor("blendList_Hidden")
	blendList_InmapFrom_ : BlendList_InmapFromPlug = PlugDescriptor("blendList_InmapFrom")
	blendList_InmapTo_ : BlendList_InmapToPlug = PlugDescriptor("blendList_InmapTo")
	blendList_Inmap_ : BlendList_InmapPlug = PlugDescriptor("blendList_Inmap")
	blendList_OutmapFrom_ : BlendList_OutmapFromPlug = PlugDescriptor("blendList_OutmapFrom")
	blendList_OutmapTo_ : BlendList_OutmapToPlug = PlugDescriptor("blendList_OutmapTo")
	blendList_Outmap_ : BlendList_OutmapPlug = PlugDescriptor("blendList_Outmap")
	blendList_Raw_ : BlendList_RawPlug = PlugDescriptor("blendList_Raw")
	blendList_ : BlendListPlug = PlugDescriptor("blendList")
	clip_ : ClipPlug = PlugDescriptor("clip")
	clipEvaluate_ : ClipEvaluatePlug = PlugDescriptor("clipEvaluate")
	clipFunction_Hidden_ : ClipFunction_HiddenPlug = PlugDescriptor("clipFunction_Hidden")
	clipFunction_InmapFrom_ : ClipFunction_InmapFromPlug = PlugDescriptor("clipFunction_InmapFrom")
	clipFunction_InmapTo_ : ClipFunction_InmapToPlug = PlugDescriptor("clipFunction_InmapTo")
	clipFunction_Inmap_ : ClipFunction_InmapPlug = PlugDescriptor("clipFunction_Inmap")
	clipFunction_OutmapFrom_ : ClipFunction_OutmapFromPlug = PlugDescriptor("clipFunction_OutmapFrom")
	clipFunction_OutmapTo_ : ClipFunction_OutmapToPlug = PlugDescriptor("clipFunction_OutmapTo")
	clipFunction_Outmap_ : ClipFunction_OutmapPlug = PlugDescriptor("clipFunction_Outmap")
	clipFunction_Raw_ : ClipFunction_RawPlug = PlugDescriptor("clipFunction_Raw")
	clipFunction_ : ClipFunctionPlug = PlugDescriptor("clipFunction")
	clipStatePercentEval_ : ClipStatePercentEvalPlug = PlugDescriptor("clipStatePercentEval")
	cycle_ : CyclePlug = PlugDescriptor("cycle")
	enable_ : EnablePlug = PlugDescriptor("enable")
	hold_ : HoldPlug = PlugDescriptor("hold")
	numTracks_ : NumTracksPlug = PlugDescriptor("numTracks")
	postCycle_ : PostCyclePlug = PlugDescriptor("postCycle")
	preCycle_ : PreCyclePlug = PlugDescriptor("preCycle")
	scale_ : ScalePlug = PlugDescriptor("scale")
	sourceEnd_ : SourceEndPlug = PlugDescriptor("sourceEnd")
	sourceStart_ : SourceStartPlug = PlugDescriptor("sourceStart")
	start_ : StartPlug = PlugDescriptor("start")
	startPercent_ : StartPercentPlug = PlugDescriptor("startPercent")
	track_ : TrackPlug = PlugDescriptor("track")
	trackState_ : TrackStatePlug = PlugDescriptor("trackState")
	weight_ : WeightPlug = PlugDescriptor("weight")
	weightStyle_ : WeightStylePlug = PlugDescriptor("weightStyle")

	# node attributes

	typeName = "clipScheduler"
	apiTypeInt = 779
	apiTypeStr = "kClipScheduler"
	typeIdInt = 1129530184
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["absolute", "absoluteRotations", "binMembership", "firstClip", "secondClip", "blendClips", "blendList_Hidden", "blendList_InmapFrom", "blendList_InmapTo", "blendList_Inmap", "blendList_OutmapFrom", "blendList_OutmapTo", "blendList_Outmap", "blendList_Raw", "blendList", "clip", "clipEvaluate", "clipFunction_Hidden", "clipFunction_InmapFrom", "clipFunction_InmapTo", "clipFunction_Inmap", "clipFunction_OutmapFrom", "clipFunction_OutmapTo", "clipFunction_Outmap", "clipFunction_Raw", "clipFunction", "clipStatePercentEval", "cycle", "enable", "hold", "numTracks", "postCycle", "preCycle", "scale", "sourceEnd", "sourceStart", "start", "startPercent", "track", "trackState", "weight", "weightStyle"]
	nodeLeafPlugs = ["absolute", "absoluteRotations", "binMembership", "blendClips", "blendList", "clip", "clipEvaluate", "clipFunction", "clipStatePercentEval", "cycle", "enable", "hold", "numTracks", "postCycle", "preCycle", "scale", "sourceEnd", "sourceStart", "start", "startPercent", "track", "trackState", "weight", "weightStyle"]
	pass

