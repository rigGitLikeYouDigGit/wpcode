

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifier = retriever.getNodeCls("PolyModifier")
assert PolyModifier
if T.TYPE_CHECKING:
	from .. import PolyModifier

# add node doc



# region plug type defs
class CleanVerticesPlug(Plug):
	node : PolyDelEdge = None
	pass
# endregion


# define node class
class PolyDelEdge(PolyModifier):
	cleanVertices_ : CleanVerticesPlug = PlugDescriptor("cleanVertices")

	# node attributes

	typeName = "polyDelEdge"
	apiTypeInt = 409
	apiTypeStr = "kPolyDelEdge"
	typeIdInt = 1346651461
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["cleanVertices"]
	nodeLeafPlugs = ["cleanVertices"]
	pass

