

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
class BaseWeightPlug(Plug):
	node : BlendFalloff = None
	pass
class BinMembershipPlug(Plug):
	node : BlendFalloff = None
	pass
class OutputWeightFunctionPlug(Plug):
	node : BlendFalloff = None
	pass
class ModePlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : BlendFalloff = None
	pass
class WeightPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : BlendFalloff = None
	pass
class WeightFunctionPlug(Plug):
	parent : TargetPlug = PlugDescriptor("target")
	node : BlendFalloff = None
	pass
class TargetPlug(Plug):
	mode_ : ModePlug = PlugDescriptor("mode")
	mod_ : ModePlug = PlugDescriptor("mode")
	weight_ : WeightPlug = PlugDescriptor("weight")
	wgt_ : WeightPlug = PlugDescriptor("weight")
	weightFunction_ : WeightFunctionPlug = PlugDescriptor("weightFunction")
	whf_ : WeightFunctionPlug = PlugDescriptor("weightFunction")
	node : BlendFalloff = None
	pass
# endregion


# define node class
class BlendFalloff(_BASE_):
	baseWeight_ : BaseWeightPlug = PlugDescriptor("baseWeight")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	outputWeightFunction_ : OutputWeightFunctionPlug = PlugDescriptor("outputWeightFunction")
	mode_ : ModePlug = PlugDescriptor("mode")
	weight_ : WeightPlug = PlugDescriptor("weight")
	weightFunction_ : WeightFunctionPlug = PlugDescriptor("weightFunction")
	target_ : TargetPlug = PlugDescriptor("target")

	# node attributes

	typeName = "blendFalloff"
	apiTypeInt = 1141
	apiTypeStr = "kBlendFalloff"
	typeIdInt = 1111904070
	MFnCls = om.MFnDependencyNode
	pass

