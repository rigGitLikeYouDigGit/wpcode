

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Locator = Catalogue.Locator
else:
	from .. import retriever
	Locator = retriever.getNodeCls("Locator")
	assert Locator

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

