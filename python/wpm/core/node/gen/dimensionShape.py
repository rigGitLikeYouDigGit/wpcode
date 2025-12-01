

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Shape = retriever.getNodeCls("Shape")
assert Shape
if T.TYPE_CHECKING:
	from .. import Shape

# add node doc



# region plug type defs

# endregion


# define node class
class DimensionShape(Shape):

	# node attributes

	typeName = "dimensionShape"
	typeIdInt = 1146308688
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

