

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Blend = retriever.getNodeCls("Blend")
assert Blend
if T.TYPE_CHECKING:
	from .. import Blend

# add node doc



# region plug type defs
class WeightPlug(Plug):
	node : BlendWeighted = None
	pass
# endregion


# define node class
class BlendWeighted(Blend):
	weight_ : WeightPlug = PlugDescriptor("weight")

	# node attributes

	typeName = "blendWeighted"
	apiTypeInt = 29
	apiTypeStr = "kBlendWeighted"
	typeIdInt = 1094863959
	MFnCls = om.MFnDependencyNode
	pass

