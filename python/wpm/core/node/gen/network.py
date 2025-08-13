

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
class AffectedByPlug(Plug):
	node : Network = None
	pass
class AffectsPlug(Plug):
	node : Network = None
	pass
class BinMembershipPlug(Plug):
	node : Network = None
	pass
# endregion


# define node class
class Network(_BASE_):
	affectedBy_ : AffectedByPlug = PlugDescriptor("affectedBy")
	affects_ : AffectsPlug = PlugDescriptor("affects")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")

	# node attributes

	typeName = "network"
	typeIdInt = 1314150219
	pass

