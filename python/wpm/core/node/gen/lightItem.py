

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	LightItemBase = Catalogue.LightItemBase
else:
	from .. import retriever
	LightItemBase = retriever.getNodeCls("LightItemBase")
	assert LightItemBase

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
	nodeLeafClassAttrs = ["light"]
	nodeLeafPlugs = ["light"]
	pass

