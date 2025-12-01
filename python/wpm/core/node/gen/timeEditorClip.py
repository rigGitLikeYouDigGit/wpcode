

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
TimeEditorClipBase = retriever.getNodeCls("TimeEditorClipBase")
assert TimeEditorClipBase
if T.TYPE_CHECKING:
	from .. import TimeEditorClipBase

# add node doc



# region plug type defs
class AnimSourcePlug(Plug):
	node : TimeEditorClip = None
	pass
class AudioFilePlug(Plug):
	node : TimeEditorClip = None
	pass
class BlendShapeTargetPlug(Plug):
	node : TimeEditorClip = None
	pass
class ClipSoloMutedPlug(Plug):
	node : TimeEditorClip = None
	pass
class ClipTrackMutedPlug(Plug):
	node : TimeEditorClip = None
	pass
class ClipWeightPlug(Plug):
	node : TimeEditorClip = None
	pass
class ContentPlug(Plug):
	node : TimeEditorClip = None
	pass
class GhostPlug(Plug):
	node : TimeEditorClip = None
	pass
class GhostColorBPlug(Plug):
	parent : GhostColorPlug = PlugDescriptor("ghostColor")
	node : TimeEditorClip = None
	pass
class GhostColorGPlug(Plug):
	parent : GhostColorPlug = PlugDescriptor("ghostColor")
	node : TimeEditorClip = None
	pass
class GhostColorRPlug(Plug):
	parent : GhostColorPlug = PlugDescriptor("ghostColor")
	node : TimeEditorClip = None
	pass
class GhostColorPlug(Plug):
	ghostColorB_ : GhostColorBPlug = PlugDescriptor("ghostColorB")
	gcb_ : GhostColorBPlug = PlugDescriptor("ghostColorB")
	ghostColorG_ : GhostColorGPlug = PlugDescriptor("ghostColorG")
	gcg_ : GhostColorGPlug = PlugDescriptor("ghostColorG")
	ghostColorR_ : GhostColorRPlug = PlugDescriptor("ghostColorR")
	gcr_ : GhostColorRPlug = PlugDescriptor("ghostColorR")
	node : TimeEditorClip = None
	pass
class GhostColorCustomPlug(Plug):
	node : TimeEditorClip = None
	pass
class GhostCountPostPlug(Plug):
	node : TimeEditorClip = None
	pass
class GhostCountPrePlug(Plug):
	node : TimeEditorClip = None
	pass
class GhostPostColorBPlug(Plug):
	parent : GhostPostColorPlug = PlugDescriptor("ghostPostColor")
	node : TimeEditorClip = None
	pass
class GhostPostColorGPlug(Plug):
	parent : GhostPostColorPlug = PlugDescriptor("ghostPostColor")
	node : TimeEditorClip = None
	pass
class GhostPostColorRPlug(Plug):
	parent : GhostPostColorPlug = PlugDescriptor("ghostPostColor")
	node : TimeEditorClip = None
	pass
class GhostPostColorPlug(Plug):
	ghostPostColorB_ : GhostPostColorBPlug = PlugDescriptor("ghostPostColorB")
	gtb_ : GhostPostColorBPlug = PlugDescriptor("ghostPostColorB")
	ghostPostColorG_ : GhostPostColorGPlug = PlugDescriptor("ghostPostColorG")
	gtg_ : GhostPostColorGPlug = PlugDescriptor("ghostPostColorG")
	ghostPostColorR_ : GhostPostColorRPlug = PlugDescriptor("ghostPostColorR")
	gtr_ : GhostPostColorRPlug = PlugDescriptor("ghostPostColorR")
	node : TimeEditorClip = None
	pass
class GhostPreColorBPlug(Plug):
	parent : GhostPreColorPlug = PlugDescriptor("ghostPreColor")
	node : TimeEditorClip = None
	pass
class GhostPreColorGPlug(Plug):
	parent : GhostPreColorPlug = PlugDescriptor("ghostPreColor")
	node : TimeEditorClip = None
	pass
class GhostPreColorRPlug(Plug):
	parent : GhostPreColorPlug = PlugDescriptor("ghostPreColor")
	node : TimeEditorClip = None
	pass
class GhostPreColorPlug(Plug):
	ghostPreColorB_ : GhostPreColorBPlug = PlugDescriptor("ghostPreColorB")
	geb_ : GhostPreColorBPlug = PlugDescriptor("ghostPreColorB")
	ghostPreColorG_ : GhostPreColorGPlug = PlugDescriptor("ghostPreColorG")
	geg_ : GhostPreColorGPlug = PlugDescriptor("ghostPreColorG")
	ghostPreColorR_ : GhostPreColorRPlug = PlugDescriptor("ghostPreColorR")
	ger_ : GhostPreColorRPlug = PlugDescriptor("ghostPreColorR")
	node : TimeEditorClip = None
	pass
class GhostRootCustomPlug(Plug):
	node : TimeEditorClip = None
	pass
class GhostRootDefaultPlug(Plug):
	node : TimeEditorClip = None
	pass
class GhostRootTargetsPlug(Plug):
	node : TimeEditorClip = None
	pass
class GhostStepSizePlug(Plug):
	node : TimeEditorClip = None
	pass
class LayerIdPlug(Plug):
	parent : LayerPlug = PlugDescriptor("layer")
	node : TimeEditorClip = None
	pass
class LayerIndexPlug(Plug):
	parent : LayerPlug = PlugDescriptor("layer")
	node : TimeEditorClip = None
	pass
class LayerModePlug(Plug):
	parent : LayerPlug = PlugDescriptor("layer")
	node : TimeEditorClip = None
	pass
class LayerMutedPlug(Plug):
	parent : LayerPlug = PlugDescriptor("layer")
	node : TimeEditorClip = None
	pass
class LayerNamePlug(Plug):
	parent : LayerPlug = PlugDescriptor("layer")
	node : TimeEditorClip = None
	pass
class LayerSoloPlug(Plug):
	parent : LayerPlug = PlugDescriptor("layer")
	node : TimeEditorClip = None
	pass
class LayerWeightPlug(Plug):
	parent : LayerPlug = PlugDescriptor("layer")
	node : TimeEditorClip = None
	pass
class LayerPlug(Plug):
	layerId_ : LayerIdPlug = PlugDescriptor("layerId")
	lid_ : LayerIdPlug = PlugDescriptor("layerId")
	layerIndex_ : LayerIndexPlug = PlugDescriptor("layerIndex")
	li_ : LayerIndexPlug = PlugDescriptor("layerIndex")
	layerMode_ : LayerModePlug = PlugDescriptor("layerMode")
	lm_ : LayerModePlug = PlugDescriptor("layerMode")
	layerMuted_ : LayerMutedPlug = PlugDescriptor("layerMuted")
	lmd_ : LayerMutedPlug = PlugDescriptor("layerMuted")
	layerName_ : LayerNamePlug = PlugDescriptor("layerName")
	ln_ : LayerNamePlug = PlugDescriptor("layerName")
	layerSolo_ : LayerSoloPlug = PlugDescriptor("layerSolo")
	lsl_ : LayerSoloPlug = PlugDescriptor("layerSolo")
	layerWeight_ : LayerWeightPlug = PlugDescriptor("layerWeight")
	lw_ : LayerWeightPlug = PlugDescriptor("layerWeight")
	node : TimeEditorClip = None
	pass
class StatePlug(Plug):
	node : TimeEditorClip = None
	pass
class TrackPlug(Plug):
	node : TimeEditorClip = None
	pass
class TransitionToPlug(Plug):
	node : TimeEditorClip = None
	pass
# endregion


# define node class
class TimeEditorClip(TimeEditorClipBase):
	animSource_ : AnimSourcePlug = PlugDescriptor("animSource")
	audioFile_ : AudioFilePlug = PlugDescriptor("audioFile")
	blendShapeTarget_ : BlendShapeTargetPlug = PlugDescriptor("blendShapeTarget")
	clipSoloMuted_ : ClipSoloMutedPlug = PlugDescriptor("clipSoloMuted")
	clipTrackMuted_ : ClipTrackMutedPlug = PlugDescriptor("clipTrackMuted")
	clipWeight_ : ClipWeightPlug = PlugDescriptor("clipWeight")
	content_ : ContentPlug = PlugDescriptor("content")
	ghost_ : GhostPlug = PlugDescriptor("ghost")
	ghostColorB_ : GhostColorBPlug = PlugDescriptor("ghostColorB")
	ghostColorG_ : GhostColorGPlug = PlugDescriptor("ghostColorG")
	ghostColorR_ : GhostColorRPlug = PlugDescriptor("ghostColorR")
	ghostColor_ : GhostColorPlug = PlugDescriptor("ghostColor")
	ghostColorCustom_ : GhostColorCustomPlug = PlugDescriptor("ghostColorCustom")
	ghostCountPost_ : GhostCountPostPlug = PlugDescriptor("ghostCountPost")
	ghostCountPre_ : GhostCountPrePlug = PlugDescriptor("ghostCountPre")
	ghostPostColorB_ : GhostPostColorBPlug = PlugDescriptor("ghostPostColorB")
	ghostPostColorG_ : GhostPostColorGPlug = PlugDescriptor("ghostPostColorG")
	ghostPostColorR_ : GhostPostColorRPlug = PlugDescriptor("ghostPostColorR")
	ghostPostColor_ : GhostPostColorPlug = PlugDescriptor("ghostPostColor")
	ghostPreColorB_ : GhostPreColorBPlug = PlugDescriptor("ghostPreColorB")
	ghostPreColorG_ : GhostPreColorGPlug = PlugDescriptor("ghostPreColorG")
	ghostPreColorR_ : GhostPreColorRPlug = PlugDescriptor("ghostPreColorR")
	ghostPreColor_ : GhostPreColorPlug = PlugDescriptor("ghostPreColor")
	ghostRootCustom_ : GhostRootCustomPlug = PlugDescriptor("ghostRootCustom")
	ghostRootDefault_ : GhostRootDefaultPlug = PlugDescriptor("ghostRootDefault")
	ghostRootTargets_ : GhostRootTargetsPlug = PlugDescriptor("ghostRootTargets")
	ghostStepSize_ : GhostStepSizePlug = PlugDescriptor("ghostStepSize")
	layerId_ : LayerIdPlug = PlugDescriptor("layerId")
	layerIndex_ : LayerIndexPlug = PlugDescriptor("layerIndex")
	layerMode_ : LayerModePlug = PlugDescriptor("layerMode")
	layerMuted_ : LayerMutedPlug = PlugDescriptor("layerMuted")
	layerName_ : LayerNamePlug = PlugDescriptor("layerName")
	layerSolo_ : LayerSoloPlug = PlugDescriptor("layerSolo")
	layerWeight_ : LayerWeightPlug = PlugDescriptor("layerWeight")
	layer_ : LayerPlug = PlugDescriptor("layer")
	state_ : StatePlug = PlugDescriptor("state")
	track_ : TrackPlug = PlugDescriptor("track")
	transitionTo_ : TransitionToPlug = PlugDescriptor("transitionTo")

	# node attributes

	typeName = "timeEditorClip"
	apiTypeInt = 1105
	apiTypeStr = "kTimeEditorClip"
	typeIdInt = 1094929475
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["animSource", "audioFile", "blendShapeTarget", "clipSoloMuted", "clipTrackMuted", "clipWeight", "content", "ghost", "ghostColorB", "ghostColorG", "ghostColorR", "ghostColor", "ghostColorCustom", "ghostCountPost", "ghostCountPre", "ghostPostColorB", "ghostPostColorG", "ghostPostColorR", "ghostPostColor", "ghostPreColorB", "ghostPreColorG", "ghostPreColorR", "ghostPreColor", "ghostRootCustom", "ghostRootDefault", "ghostRootTargets", "ghostStepSize", "layerId", "layerIndex", "layerMode", "layerMuted", "layerName", "layerSolo", "layerWeight", "layer", "state", "track", "transitionTo"]
	nodeLeafPlugs = ["animSource", "audioFile", "blendShapeTarget", "clipSoloMuted", "clipTrackMuted", "clipWeight", "content", "ghost", "ghostColor", "ghostColorCustom", "ghostCountPost", "ghostCountPre", "ghostPostColor", "ghostPreColor", "ghostRootCustom", "ghostRootDefault", "ghostRootTargets", "ghostStepSize", "layer", "state", "track", "transitionTo"]
	pass

