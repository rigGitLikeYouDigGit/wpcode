

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
	node : ComponentFalloff = None
	pass
class OutputWeightFunctionPlug(Plug):
	node : ComponentFalloff = None
	pass
class DefaultWeightPlug(Plug):
	parent : WeightInfoLayersPlug = PlugDescriptor("weightInfoLayers")
	node : ComponentFalloff = None
	pass
class LayerNamePlug(Plug):
	parent : WeightInfoLayersPlug = PlugDescriptor("weightInfoLayers")
	node : ComponentFalloff = None
	pass
class WeightInfoLayersPlug(Plug):
	defaultWeight_ : DefaultWeightPlug = PlugDescriptor("defaultWeight")
	dwt_ : DefaultWeightPlug = PlugDescriptor("defaultWeight")
	layerName_ : LayerNamePlug = PlugDescriptor("layerName")
	lnm_ : LayerNamePlug = PlugDescriptor("layerName")
	node : ComponentFalloff = None
	pass
class WeightsPlug(Plug):
	parent : WeightLayersPlug = PlugDescriptor("weightLayers")
	node : ComponentFalloff = None
	pass
class WeightLayersPlug(Plug):
	weights_ : WeightsPlug = PlugDescriptor("weights")
	wht_ : WeightsPlug = PlugDescriptor("weights")
	node : ComponentFalloff = None
	pass
class WeightedGeometryPlug(Plug):
	node : ComponentFalloff = None
	pass
# endregion


# define node class
class ComponentFalloff(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	outputWeightFunction_ : OutputWeightFunctionPlug = PlugDescriptor("outputWeightFunction")
	defaultWeight_ : DefaultWeightPlug = PlugDescriptor("defaultWeight")
	layerName_ : LayerNamePlug = PlugDescriptor("layerName")
	weightInfoLayers_ : WeightInfoLayersPlug = PlugDescriptor("weightInfoLayers")
	weights_ : WeightsPlug = PlugDescriptor("weights")
	weightLayers_ : WeightLayersPlug = PlugDescriptor("weightLayers")
	weightedGeometry_ : WeightedGeometryPlug = PlugDescriptor("weightedGeometry")

	# node attributes

	typeName = "componentFalloff"
	apiTypeInt = 1144
	apiTypeStr = "kComponentFalloff"
	typeIdInt = 1195790150
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "outputWeightFunction", "defaultWeight", "layerName", "weightInfoLayers", "weights", "weightLayers", "weightedGeometry"]
	nodeLeafPlugs = ["binMembership", "outputWeightFunction", "weightInfoLayers", "weightLayers", "weightedGeometry"]
	pass

