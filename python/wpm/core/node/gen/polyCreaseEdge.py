

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyCrease = retriever.getNodeCls("PolyCrease")
assert PolyCrease
if T.TYPE_CHECKING:
	from .. import PolyCrease

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

