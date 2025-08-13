

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
class LightPlug(Plug):
	node : LightItem = None
	pass
# endregion


# define node class
class LightItem(LightItemBase):
	light_ : LightPlug = PlugDescriptor("light")

	# node attributes

	typeName = "lightItem"
	typeIdInt = 1476396001
	pass

