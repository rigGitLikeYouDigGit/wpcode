

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
	node : GlobalCacheControl = None
	pass
class EnableStatusPlug(Plug):
	node : GlobalCacheControl = None
	pass
class WriteEnablePlug(Plug):
	node : GlobalCacheControl = None
	pass
# endregion


# define node class
class GlobalCacheControl(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	enableStatus_ : EnableStatusPlug = PlugDescriptor("enableStatus")
	writeEnable_ : WriteEnablePlug = PlugDescriptor("writeEnable")

	# node attributes

	typeName = "globalCacheControl"
	typeIdInt = 1195590476
	nodeLeafClassAttrs = ["binMembership", "enableStatus", "writeEnable"]
	nodeLeafPlugs = ["binMembership", "enableStatus", "writeEnable"]
	pass

