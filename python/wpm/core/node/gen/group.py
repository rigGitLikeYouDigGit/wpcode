

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	RScontainer = Catalogue.RScontainer
else:
	from .. import retriever
	RScontainer = retriever.getNodeCls("RScontainer")
	assert RScontainer

# add node doc



# region plug type defs

# endregion


# define node class
class Group(RScontainer):

	# node attributes

	typeName = "group"
	typeIdInt = 1476395941
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

