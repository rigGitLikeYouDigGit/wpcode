

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
	node : GroupId = None
	pass
class GroupIdPlug(Plug):
	node : GroupId = None
	pass
# endregion


# define node class
class GroupId(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	groupId_ : GroupIdPlug = PlugDescriptor("groupId")

	# node attributes

	typeName = "groupId"
	apiTypeInt = 356
	apiTypeStr = "kGroupId"
	typeIdInt = 1196443972
	MFnCls = om.MFnDependencyNode
	pass

