

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
class BinMembershipPlug(Plug):
	node : TransferFalloff = None
	pass
class BindTagsFilterPlug(Plug):
	node : TransferFalloff = None
	pass
class CacheSetupPlug(Plug):
	node : TransferFalloff = None
	pass
class OutputWeightFunctionPlug(Plug):
	node : TransferFalloff = None
	pass
class UseBindTagsPlug(Plug):
	node : TransferFalloff = None
	pass
class WeightFunctionPlug(Plug):
	node : TransferFalloff = None
	pass
class WeightedGeometryPlug(Plug):
	node : TransferFalloff = None
	pass
# endregion


# define node class
class TransferFalloff(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	bindTagsFilter_ : BindTagsFilterPlug = PlugDescriptor("bindTagsFilter")
	cacheSetup_ : CacheSetupPlug = PlugDescriptor("cacheSetup")
	outputWeightFunction_ : OutputWeightFunctionPlug = PlugDescriptor("outputWeightFunction")
	useBindTags_ : UseBindTagsPlug = PlugDescriptor("useBindTags")
	weightFunction_ : WeightFunctionPlug = PlugDescriptor("weightFunction")
	weightedGeometry_ : WeightedGeometryPlug = PlugDescriptor("weightedGeometry")

	# node attributes

	typeName = "transferFalloff"
	apiTypeInt = 1143
	apiTypeStr = "kTransferFalloff"
	typeIdInt = 1346785094
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "bindTagsFilter", "cacheSetup", "outputWeightFunction", "useBindTags", "weightFunction", "weightedGeometry"]
	nodeLeafPlugs = ["binMembership", "bindTagsFilter", "cacheSetup", "outputWeightFunction", "useBindTags", "weightFunction", "weightedGeometry"]
	pass

