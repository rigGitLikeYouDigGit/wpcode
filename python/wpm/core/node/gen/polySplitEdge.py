

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
class OperationPlug(Plug):
	node : PolySplitEdge = None
	pass
# endregion


# define node class
class PolySplitEdge(PolyModifier):
	operation_ : OperationPlug = PlugDescriptor("operation")

	# node attributes

	typeName = "polySplitEdge"
	apiTypeInt = 815
	apiTypeStr = "kPolySplitEdge"
	typeIdInt = 1347634500
	MFnCls = om.MFnDependencyNode
	pass

