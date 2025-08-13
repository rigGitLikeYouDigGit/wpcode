

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
	node : ListItem = None
	pass
class NextPlug(Plug):
	node : ListItem = None
	pass
class ParentListPlug(Plug):
	node : ListItem = None
	pass
class PreviousPlug(Plug):
	node : ListItem = None
	pass
# endregion


# define node class
class ListItem(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	next_ : NextPlug = PlugDescriptor("next")
	parentList_ : ParentListPlug = PlugDescriptor("parentList")
	previous_ : PreviousPlug = PlugDescriptor("previous")

	# node attributes

	typeName = "listItem"
	typeIdInt = 1476395894
	pass

