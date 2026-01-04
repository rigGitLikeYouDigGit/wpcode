

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	GeometryShape = Catalogue.GeometryShape
else:
	from .. import retriever
	GeometryShape = retriever.getNodeCls("GeometryShape")
	assert GeometryShape

# add node doc



# region plug type defs

# endregion


# define node class
class Plane(GeometryShape):

	# node attributes

	typeName = "plane"
	typeIdInt = 1347178053
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

