

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
	node : UniformFalloff = None
	pass
class OutputWeightFunctionPlug(Plug):
	node : UniformFalloff = None
	pass
class UniformWeightPlug(Plug):
	node : UniformFalloff = None
	pass
# endregion


# define node class
class UniformFalloff(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	outputWeightFunction_ : OutputWeightFunctionPlug = PlugDescriptor("outputWeightFunction")
	uniformWeight_ : UniformWeightPlug = PlugDescriptor("uniformWeight")

	# node attributes

	typeName = "uniformFalloff"
	apiTypeInt = 1142
	apiTypeStr = "kUniformFalloff"
	typeIdInt = 1431783252
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "outputWeightFunction", "uniformWeight"]
	nodeLeafPlugs = ["binMembership", "outputWeightFunction", "uniformWeight"]
	pass

