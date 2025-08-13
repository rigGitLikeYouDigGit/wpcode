

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
	node : CacheBase = None
	pass
class InRangePlug(Plug):
	node : CacheBase = None
	pass
class OutCacheArrayDataPlug(Plug):
	node : CacheBase = None
	pass
class OutCacheDataPlug(Plug):
	node : CacheBase = None
	pass
# endregion


# define node class
class CacheBase(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	inRange_ : InRangePlug = PlugDescriptor("inRange")
	outCacheArrayData_ : OutCacheArrayDataPlug = PlugDescriptor("outCacheArrayData")
	outCacheData_ : OutCacheDataPlug = PlugDescriptor("outCacheData")

	# node attributes

	typeName = "cacheBase"
	apiTypeInt = 999
	apiTypeStr = "kCacheBase"
	typeIdInt = 1128415571
	MFnCls = om.MFnDependencyNode
	pass

