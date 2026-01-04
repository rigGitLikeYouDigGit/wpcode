

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyCrease = Catalogue.PolyCrease
else:
	from .. import retriever
	PolyCrease = retriever.getNodeCls("PolyCrease")
	assert PolyCrease

# add node doc



# region plug type defs

# endregion


# define node class
class PolyCreaseEdge(PolyCrease):

	# node attributes

	typeName = "polyCreaseEdge"
	apiTypeInt = 959
	apiTypeStr = "kPolyCreaseEdge"
	typeIdInt = 1346589509
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

