

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ObjectSet = Catalogue.ObjectSet
else:
	from .. import retriever
	ObjectSet = retriever.getNodeCls("ObjectSet")
	assert ObjectSet

# add node doc



# region plug type defs
class BackgroundWeightPlug(Plug):
	node : AnimLayer = None
	pass
class BlendNodesPlug(Plug):
	node : AnimLayer = None
	pass
class ChildrenLayersPlug(Plug):
	node : AnimLayer = None
	pass
class ChildrenSoloPlug(Plug):
	node : AnimLayer = None
	pass
class ChildsoloedPlug(Plug):
	node : AnimLayer = None
	pass
class ClipsPlug(Plug):
	node : AnimLayer = None
	pass
class CollapsePlug(Plug):
	node : AnimLayer = None
	pass
class CteRootPlug(Plug):
	node : AnimLayer = None
	pass
class ExclusivePlug(Plug):
	node : AnimLayer = None
	pass
class ForegroundWeightPlug(Plug):
	node : AnimLayer = None
	pass
class GhostPlug(Plug):
	node : AnimLayer = None
	pass
class GhostColorPlug(Plug):
	node : AnimLayer = None
	pass
class GhostedClipsPlug(Plug):
	node : AnimLayer = None
	pass
class LockPlug(Plug):
	node : AnimLayer = None
	pass
class MutePlug(Plug):
	node : AnimLayer = None
	pass
class OutMutePlug(Plug):
	node : AnimLayer = None
	pass
class OutRotationAccumulationModePlug(Plug):
	node : AnimLayer = None
	pass
class OverridePlug(Plug):
	node : AnimLayer = None
	pass
class ParentLayerPlug(Plug):
	node : AnimLayer = None
	pass
class ParentMutePlug(Plug):
	node : AnimLayer = None
	pass
class ParentWeightPlug(Plug):
	node : AnimLayer = None
	pass
class PassthroughPlug(Plug):
	node : AnimLayer = None
	pass
class PreferredPlug(Plug):
	node : AnimLayer = None
	pass
class RotationAccumulationModePlug(Plug):
	node : AnimLayer = None
	pass
class ScaleAccumulationModePlug(Plug):
	node : AnimLayer = None
	pass
class SelectedPlug(Plug):
	node : AnimLayer = None
	pass
class SiblingSoloPlug(Plug):
	node : AnimLayer = None
	pass
class SoloPlug(Plug):
	node : AnimLayer = None
	pass
class WeightPlug(Plug):
	node : AnimLayer = None
	pass
# endregion


# define node class
class AnimLayer(ObjectSet):
	backgroundWeight_ : BackgroundWeightPlug = PlugDescriptor("backgroundWeight")
	blendNodes_ : BlendNodesPlug = PlugDescriptor("blendNodes")
	childrenLayers_ : ChildrenLayersPlug = PlugDescriptor("childrenLayers")
	childrenSolo_ : ChildrenSoloPlug = PlugDescriptor("childrenSolo")
	childsoloed_ : ChildsoloedPlug = PlugDescriptor("childsoloed")
	clips_ : ClipsPlug = PlugDescriptor("clips")
	collapse_ : CollapsePlug = PlugDescriptor("collapse")
	cteRoot_ : CteRootPlug = PlugDescriptor("cteRoot")
	exclusive_ : ExclusivePlug = PlugDescriptor("exclusive")
	foregroundWeight_ : ForegroundWeightPlug = PlugDescriptor("foregroundWeight")
	ghost_ : GhostPlug = PlugDescriptor("ghost")
	ghostColor_ : GhostColorPlug = PlugDescriptor("ghostColor")
	ghostedClips_ : GhostedClipsPlug = PlugDescriptor("ghostedClips")
	lock_ : LockPlug = PlugDescriptor("lock")
	mute_ : MutePlug = PlugDescriptor("mute")
	outMute_ : OutMutePlug = PlugDescriptor("outMute")
	outRotationAccumulationMode_ : OutRotationAccumulationModePlug = PlugDescriptor("outRotationAccumulationMode")
	override_ : OverridePlug = PlugDescriptor("override")
	parentLayer_ : ParentLayerPlug = PlugDescriptor("parentLayer")
	parentMute_ : ParentMutePlug = PlugDescriptor("parentMute")
	parentWeight_ : ParentWeightPlug = PlugDescriptor("parentWeight")
	passthrough_ : PassthroughPlug = PlugDescriptor("passthrough")
	preferred_ : PreferredPlug = PlugDescriptor("preferred")
	rotationAccumulationMode_ : RotationAccumulationModePlug = PlugDescriptor("rotationAccumulationMode")
	scaleAccumulationMode_ : ScaleAccumulationModePlug = PlugDescriptor("scaleAccumulationMode")
	selected_ : SelectedPlug = PlugDescriptor("selected")
	siblingSolo_ : SiblingSoloPlug = PlugDescriptor("siblingSolo")
	solo_ : SoloPlug = PlugDescriptor("solo")
	weight_ : WeightPlug = PlugDescriptor("weight")

	# node attributes

	typeName = "animLayer"
	apiTypeInt = 1020
	apiTypeStr = "kAnimLayer"
	typeIdInt = 1095650386
	MFnCls = om.MFnSet
	nodeLeafClassAttrs = ["backgroundWeight", "blendNodes", "childrenLayers", "childrenSolo", "childsoloed", "clips", "collapse", "cteRoot", "exclusive", "foregroundWeight", "ghost", "ghostColor", "ghostedClips", "lock", "mute", "outMute", "outRotationAccumulationMode", "override", "parentLayer", "parentMute", "parentWeight", "passthrough", "preferred", "rotationAccumulationMode", "scaleAccumulationMode", "selected", "siblingSolo", "solo", "weight"]
	nodeLeafPlugs = ["backgroundWeight", "blendNodes", "childrenLayers", "childrenSolo", "childsoloed", "clips", "collapse", "cteRoot", "exclusive", "foregroundWeight", "ghost", "ghostColor", "ghostedClips", "lock", "mute", "outMute", "outRotationAccumulationMode", "override", "parentLayer", "parentMute", "parentWeight", "passthrough", "preferred", "rotationAccumulationMode", "scaleAccumulationMode", "selected", "siblingSolo", "solo", "weight"]
	pass

