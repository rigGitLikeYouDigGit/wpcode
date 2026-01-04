

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifier = Catalogue.PolyModifier
else:
	from .. import retriever
	PolyModifier = retriever.getNodeCls("PolyModifier")
	assert PolyModifier

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

