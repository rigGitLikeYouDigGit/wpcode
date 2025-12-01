

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Collection = retriever.getNodeCls("Collection")
assert Collection
if T.TYPE_CHECKING:
	from .. import Collection

# add node doc



# region plug type defs

# endregion


# define node class
class LightsCollection(Collection):

	# node attributes

	typeName = "lightsCollection"
	typeIdInt = 1476395924
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

