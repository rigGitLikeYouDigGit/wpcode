

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
class TexturePlug(Plug):
	node : PolySewEdge = None
	pass
class TolerancePlug(Plug):
	node : PolySewEdge = None
	pass
# endregion


# define node class
class PolySewEdge(PolyModifierWorld):
	texture_ : TexturePlug = PlugDescriptor("texture")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")

	# node attributes

	typeName = "polySewEdge"
	apiTypeInt = 697
	apiTypeStr = "kPolySewEdge"
	typeIdInt = 1347639109
	MFnCls = om.MFnDependencyNode
	pass

