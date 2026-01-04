

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	SurfaceShape = Catalogue.SurfaceShape
else:
	from .. import retriever
	SurfaceShape = retriever.getNodeCls("SurfaceShape")
	assert SurfaceShape

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

