

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifierWorld = retriever.getNodeCls("PolyModifierWorld")
assert PolyModifierWorld
if T.TYPE_CHECKING:
	from .. import PolyModifierWorld

# add node doc



# region plug type defs
class AnglePlug(Plug):
	node : PolySoftEdge = None
	pass
# endregion


# define node class
class PolySoftEdge(PolyModifierWorld):
	angle_ : AnglePlug = PlugDescriptor("angle")

	# node attributes

	typeName = "polySoftEdge"
	apiTypeInt = 429
	apiTypeStr = "kPolySoftEdge"
	typeIdInt = 1347637061
	MFnCls = om.MFnDependencyNode
	pass

