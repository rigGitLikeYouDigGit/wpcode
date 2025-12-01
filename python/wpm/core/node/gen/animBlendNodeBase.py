

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
	node : AnimBlendNodeBase = None
	pass
class DestinationPlugPlug(Plug):
	node : AnimBlendNodeBase = None
	pass
class WeightAPlug(Plug):
	node : AnimBlendNodeBase = None
	pass
class WeightBPlug(Plug):
	node : AnimBlendNodeBase = None
	pass
# endregion


# define node class
class AnimBlendNodeBase(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	destinationPlug_ : DestinationPlugPlug = PlugDescriptor("destinationPlug")
	weightA_ : WeightAPlug = PlugDescriptor("weightA")
	weightB_ : WeightBPlug = PlugDescriptor("weightB")

	# node attributes

	typeName = "animBlendNodeBase"
	typeIdInt = 1094864450
	nodeLeafClassAttrs = ["binMembership", "destinationPlug", "weightA", "weightB"]
	nodeLeafPlugs = ["binMembership", "destinationPlug", "weightA", "weightB"]
	pass

