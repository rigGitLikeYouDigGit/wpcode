

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ListItem = Catalogue.ListItem
else:
	from .. import retriever
	ListItem = retriever.getNodeCls("ListItem")
	assert ListItem

# add node doc



# region plug type defs

# endregion


# define node class
class ChildNode(ListItem):

	# node attributes

	typeName = "childNode"
	typeIdInt = 1476395917
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

