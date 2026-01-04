

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
class InputPlug(Plug):
	parent : AttributePlug = PlugDescriptor("attribute")
	node : TimeEditorClipEvaluator = None
	pass
class SourcePlug(Plug):
	parent : AttributePlug = PlugDescriptor("attribute")
	node : TimeEditorClipEvaluator = None
	pass
class StartPlug(Plug):
	parent : AttributePlug = PlugDescriptor("attribute")
	node : TimeEditorClipEvaluator = None
	pass
class SwitcherPlug(Plug):
	parent : AttributePlug = PlugDescriptor("attribute")
	node : TimeEditorClipEvaluator = None
	pass
class ValuePlug(Plug):
	parent : AttributePlug = PlugDescriptor("attribute")
	node : TimeEditorClipEvaluator = None
	pass
class AttributePlug(Plug):
	input_ : InputPlug = PlugDescriptor("input")
	ai_ : InputPlug = PlugDescriptor("input")
	source_ : SourcePlug = PlugDescriptor("source")
	src_ : SourcePlug = PlugDescriptor("source")
	start_ : StartPlug = PlugDescriptor("start")
	as_ : StartPlug = PlugDescriptor("start")
	switcher_ : SwitcherPlug = PlugDescriptor("switcher")
	sw_ : SwitcherPlug = PlugDescriptor("switcher")
	value_ : ValuePlug = PlugDescriptor("value")
	av_ : ValuePlug = PlugDescriptor("value")
	node : TimeEditorClipEvaluator = None
	pass
class BinMembershipPlug(Plug):
	node : TimeEditorClipEvaluator = None
	pass
class LayerAttributeIndexPlug(Plug):
	parent : LayerAttributePlug = PlugDescriptor("layerAttribute")
	node : TimeEditorClipEvaluator = None
	pass
class LayerAttributeInputPlug(Plug):
	parent : LayerAttributePlug = PlugDescriptor("layerAttribute")
	node : TimeEditorClipEvaluator = None
	pass
class LayerAttributeLayerIdPlug(Plug):
	parent : LayerAttributePlug = PlugDescriptor("layerAttribute")
	node : TimeEditorClipEvaluator = None
	pass
class LayerAttributeValuePlug(Plug):
	parent : LayerAttributePlug = PlugDescriptor("layerAttribute")
	node : TimeEditorClipEvaluator = None
	pass
class LayerAttributePlug(Plug):
	layerAttributeIndex_ : LayerAttributeIndexPlug = PlugDescriptor("layerAttributeIndex")
	lai_ : LayerAttributeIndexPlug = PlugDescriptor("layerAttributeIndex")
	layerAttributeInput_ : LayerAttributeInputPlug = PlugDescriptor("layerAttributeInput")
	lin_ : LayerAttributeInputPlug = PlugDescriptor("layerAttributeInput")
	layerAttributeLayerId_ : LayerAttributeLayerIdPlug = PlugDescriptor("layerAttributeLayerId")
	lid_ : LayerAttributeLayerIdPlug = PlugDescriptor("layerAttributeLayerId")
	layerAttributeValue_ : LayerAttributeValuePlug = PlugDescriptor("layerAttributeValue")
	lv_ : LayerAttributeValuePlug = PlugDescriptor("layerAttributeValue")
	node : TimeEditorClipEvaluator = None
	pass
class OutputPlug(Plug):
	node : TimeEditorClipEvaluator = None
	pass
class ParentContainerStatePlug(Plug):
	node : TimeEditorClipEvaluator = None
	pass
class RosterItemsPlug(Plug):
	node : TimeEditorClipEvaluator = None
	pass
# endregion


# define node class
class TimeEditorClipEvaluator(_BASE_):
	input_ : InputPlug = PlugDescriptor("input")
	source_ : SourcePlug = PlugDescriptor("source")
	start_ : StartPlug = PlugDescriptor("start")
	switcher_ : SwitcherPlug = PlugDescriptor("switcher")
	value_ : ValuePlug = PlugDescriptor("value")
	attribute_ : AttributePlug = PlugDescriptor("attribute")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	layerAttributeIndex_ : LayerAttributeIndexPlug = PlugDescriptor("layerAttributeIndex")
	layerAttributeInput_ : LayerAttributeInputPlug = PlugDescriptor("layerAttributeInput")
	layerAttributeLayerId_ : LayerAttributeLayerIdPlug = PlugDescriptor("layerAttributeLayerId")
	layerAttributeValue_ : LayerAttributeValuePlug = PlugDescriptor("layerAttributeValue")
	layerAttribute_ : LayerAttributePlug = PlugDescriptor("layerAttribute")
	output_ : OutputPlug = PlugDescriptor("output")
	parentContainerState_ : ParentContainerStatePlug = PlugDescriptor("parentContainerState")
	rosterItems_ : RosterItemsPlug = PlugDescriptor("rosterItems")

	# node attributes

	typeName = "timeEditorClipEvaluator"
	apiTypeInt = 1104
	apiTypeStr = "kTimeEditorClipEvaluator"
	typeIdInt = 1094931023
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["input", "source", "start", "switcher", "value", "attribute", "binMembership", "layerAttributeIndex", "layerAttributeInput", "layerAttributeLayerId", "layerAttributeValue", "layerAttribute", "output", "parentContainerState", "rosterItems"]
	nodeLeafPlugs = ["attribute", "binMembership", "layerAttribute", "output", "parentContainerState", "rosterItems"]
	pass

