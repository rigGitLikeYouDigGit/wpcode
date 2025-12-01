

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
GeometryShape = retriever.getNodeCls("GeometryShape")
assert GeometryShape
if T.TYPE_CHECKING:
	from .. import GeometryShape

# add node doc



# region plug type defs

# endregion


# define node class
class EnvironmentFog(GeometryShape):

	# node attributes

	typeName = "environmentFog"
	typeIdInt = 1162761798
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

