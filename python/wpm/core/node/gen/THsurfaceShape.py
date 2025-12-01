

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
SurfaceShape = retriever.getNodeCls("SurfaceShape")
assert SurfaceShape
if T.TYPE_CHECKING:
	from .. import SurfaceShape

# add node doc



# region plug type defs

# endregion


# define node class
class THsurfaceShape(SurfaceShape):

	# node attributes

	typeName = "THsurfaceShape"
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

