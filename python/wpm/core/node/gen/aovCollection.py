

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Collection = Catalogue.Collection
else:
	from .. import retriever
	Collection = retriever.getNodeCls("Collection")
	assert Collection

# add node doc



# region plug type defs

# endregion


# define node class
class AovCollection(Collection):

	# node attributes

	typeName = "aovCollection"
	typeIdInt = 1476395931
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

