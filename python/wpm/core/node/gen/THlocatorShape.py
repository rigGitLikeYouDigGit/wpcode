

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Locator = retriever.getNodeCls("Locator")
assert Locator
if T.TYPE_CHECKING:
	from .. import Locator

# add node doc



# region plug type defs

# endregion


# define node class
class THlocatorShape(Locator):

	# node attributes

	typeName = "THlocatorShape"
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

