

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyBase = retriever.getNodeCls("PolyBase")
assert PolyBase
if T.TYPE_CHECKING:
	from .. import PolyBase

# add node doc



# region plug type defs

# endregion


# define node class
class PolyCreator(PolyBase):

	# node attributes

	typeName = "polyCreator"
	typeIdInt = 1346580014
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

