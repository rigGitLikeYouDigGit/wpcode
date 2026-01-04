

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
	node : TimeEditorInterpolator = None
	pass
class InputPlug(Plug):
	node : TimeEditorInterpolator = None
	pass
class LayerAttributeIndexPlug(Plug):
	parent : LayerAttributePlug = PlugDescriptor("layerAttribute")
	node : TimeEditorInterpolator = None
	pass
class LayerAttributeInputPlug(Plug):
	parent : LayerAttributePlug = PlugDescriptor("layerAttribute")
	node : TimeEditorInterpolator = None
	pass
class LayerAttributeLayerIdPlug(Plug):
	parent : LayerAttributePlug = PlugDescriptor("layerAttribute")
	node : TimeEditorInterpolator = None
	pass
class LayerAttributeValuePlug(Plug):
	parent : LayerAttributePlug = PlugDescriptor("layerAttribute")
	node : TimeEditorInterpolator = None
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
	node : TimeEditorInterpolator = None
	pass
class OutputPlug(Plug):
	node : TimeEditorInterpolator = None
	pass
class OutputRawPlug(Plug):
	node : TimeEditorInterpolator = None
	pass
class ParentCompoundPlug(Plug):
	node : TimeEditorInterpolator = None
	pass
class ParentCompoundStatePlug(Plug):
	node : TimeEditorInterpolator = None
	pass
class ParentTracksStatePlug(Plug):
	node : TimeEditorInterpolator = None
	pass
class TargetAttributePlug(Plug):
	node : TimeEditorInterpolator = None
	pass
# endregion


# define node class
class TimeEditorInterpolator(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	input_ : InputPlug = PlugDescriptor("input")
	layerAttributeIndex_ : LayerAttributeIndexPlug = PlugDescriptor("layerAttributeIndex")
	layerAttributeInput_ : LayerAttributeInputPlug = PlugDescriptor("layerAttributeInput")
	layerAttributeLayerId_ : LayerAttributeLayerIdPlug = PlugDescriptor("layerAttributeLayerId")
	layerAttributeValue_ : LayerAttributeValuePlug = PlugDescriptor("layerAttributeValue")
	layerAttribute_ : LayerAttributePlug = PlugDescriptor("layerAttribute")
	output_ : OutputPlug = PlugDescriptor("output")
	outputRaw_ : OutputRawPlug = PlugDescriptor("outputRaw")
	parentCompound_ : ParentCompoundPlug = PlugDescriptor("parentCompound")
	parentCompoundState_ : ParentCompoundStatePlug = PlugDescriptor("parentCompoundState")
	parentTracksState_ : ParentTracksStatePlug = PlugDescriptor("parentTracksState")
	targetAttribute_ : TargetAttributePlug = PlugDescriptor("targetAttribute")

	# node attributes

	typeName = "timeEditorInterpolator"
	apiTypeInt = 1108
	apiTypeStr = "kTimeEditorInterpolator"
	typeIdInt = 1413826896
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "input", "layerAttributeIndex", "layerAttributeInput", "layerAttributeLayerId", "layerAttributeValue", "layerAttribute", "output", "outputRaw", "parentCompound", "parentCompoundState", "parentTracksState", "targetAttribute"]
	nodeLeafPlugs = ["binMembership", "input", "layerAttribute", "output", "outputRaw", "parentCompound", "parentCompoundState", "parentTracksState", "targetAttribute"]
	pass

