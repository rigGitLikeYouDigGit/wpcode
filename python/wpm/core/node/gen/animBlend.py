

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
	node : AnimBlend = None
	pass
class BlendPlug(Plug):
	node : AnimBlend = None
	pass
class WeightPlug(Plug):
	node : AnimBlend = None
	pass
# endregion


# define node class
class AnimBlend(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	blend_ : BlendPlug = PlugDescriptor("blend")
	weight_ : WeightPlug = PlugDescriptor("weight")

	# node attributes

	typeName = "animBlend"
	apiTypeInt = 794
	apiTypeStr = "kAnimBlend"
	typeIdInt = 1094864452
	MFnCls = om.MFnDependencyNode
	pass

