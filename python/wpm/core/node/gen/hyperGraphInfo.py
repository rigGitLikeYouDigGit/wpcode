

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
	node : HyperGraphInfo = None
	pass
class BookmarksPlug(Plug):
	node : HyperGraphInfo = None
	pass
# endregion


# define node class
class HyperGraphInfo(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	bookmarks_ : BookmarksPlug = PlugDescriptor("bookmarks")

	# node attributes

	typeName = "hyperGraphInfo"
	apiTypeInt = 360
	apiTypeStr = "kHyperGraphInfo"
	typeIdInt = 1213812818
	MFnCls = om.MFnDependencyNode
	pass

