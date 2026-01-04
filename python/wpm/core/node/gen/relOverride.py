

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ValueOverride = Catalogue.ValueOverride
else:
	from .. import retriever
	ValueOverride = retriever.getNodeCls("ValueOverride")
	assert ValueOverride

# add node doc



# region plug type defs

# endregion


# define node class
class RelOverride(ValueOverride):

	# node attributes

	typeName = "relOverride"
	typeIdInt = 1476395898
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

