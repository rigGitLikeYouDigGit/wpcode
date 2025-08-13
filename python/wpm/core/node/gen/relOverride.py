

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ValueOverride = retriever.getNodeCls("ValueOverride")
assert ValueOverride
if T.TYPE_CHECKING:
	from .. import ValueOverride

# add node doc



# region plug type defs

# endregion


# define node class
class RelOverride(ValueOverride):

	# node attributes

	typeName = "relOverride"
	typeIdInt = 1476395898
	pass

