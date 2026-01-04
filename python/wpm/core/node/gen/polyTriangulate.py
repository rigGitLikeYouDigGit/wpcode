

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
class Maya80Plug(Plug):
	node : PolyTriangulate = None
	pass
# endregion


# define node class
class PolyTriangulate(PolyModifier):
	maya80_ : Maya80Plug = PlugDescriptor("maya80")

	# node attributes

	typeName = "polyTriangulate"
	apiTypeInt = 434
	apiTypeStr = "kPolyTriangulate"
	typeIdInt = 1347703369
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["maya80"]
	nodeLeafPlugs = ["maya80"]
	pass

