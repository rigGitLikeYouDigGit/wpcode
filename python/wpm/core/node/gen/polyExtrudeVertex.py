

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
class DivisionsPlug(Plug):
	node : PolyExtrudeVertex = None
	pass
class LengthPlug(Plug):
	node : PolyExtrudeVertex = None
	pass
class WidthPlug(Plug):
	node : PolyExtrudeVertex = None
	pass
# endregion


# define node class
class PolyExtrudeVertex(PolyModifierWorld):
	divisions_ : DivisionsPlug = PlugDescriptor("divisions")
	length_ : LengthPlug = PlugDescriptor("length")
	width_ : WidthPlug = PlugDescriptor("width")

	# node attributes

	typeName = "polyExtrudeVertex"
	apiTypeInt = 926
	apiTypeStr = "kPolyExtrudeVertex"
	typeIdInt = 1346721878
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["divisions", "length", "width"]
	nodeLeafPlugs = ["divisions", "length", "width"]
	pass

