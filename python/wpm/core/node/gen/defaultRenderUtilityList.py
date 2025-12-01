

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
	node : DefaultRenderUtilityList = None
	pass
class UtilitiesPlug(Plug):
	node : DefaultRenderUtilityList = None
	pass
# endregion


# define node class
class DefaultRenderUtilityList(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	utilities_ : UtilitiesPlug = PlugDescriptor("utilities")

	# node attributes

	typeName = "defaultRenderUtilityList"
	typeIdInt = 1146246476
	nodeLeafClassAttrs = ["binMembership", "utilities"]
	nodeLeafPlugs = ["binMembership", "utilities"]
	pass

