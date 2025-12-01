

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
LightItemBase = retriever.getNodeCls("LightItemBase")
assert LightItemBase
if T.TYPE_CHECKING:
	from .. import LightItemBase

# add node doc



# region plug type defs
class FirstItemPlug(Plug):
	node : LightGroup = None
	pass
class LastItemPlug(Plug):
	node : LightGroup = None
	pass
class ListItemsPlug(Plug):
	node : LightGroup = None
	pass
# endregion


# define node class
class LightGroup(LightItemBase):
	firstItem_ : FirstItemPlug = PlugDescriptor("firstItem")
	lastItem_ : LastItemPlug = PlugDescriptor("lastItem")
	listItems_ : ListItemsPlug = PlugDescriptor("listItems")

	# node attributes

	typeName = "lightGroup"
	typeIdInt = 1476396002
	nodeLeafClassAttrs = ["firstItem", "lastItem", "listItems"]
	nodeLeafPlugs = ["firstItem", "lastItem", "listItems"]
	pass

