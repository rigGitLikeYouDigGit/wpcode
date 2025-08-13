

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
class OtherPolyPlug(Plug):
	node : PolyTransfer = None
	pass
class UvSetsPlug(Plug):
	node : PolyTransfer = None
	pass
class VertexColorPlug(Plug):
	node : PolyTransfer = None
	pass
class VerticesPlug(Plug):
	node : PolyTransfer = None
	pass
# endregion


# define node class
class PolyTransfer(PolyModifier):
	otherPoly_ : OtherPolyPlug = PlugDescriptor("otherPoly")
	uvSets_ : UvSetsPlug = PlugDescriptor("uvSets")
	vertexColor_ : VertexColorPlug = PlugDescriptor("vertexColor")
	vertices_ : VerticesPlug = PlugDescriptor("vertices")

	# node attributes

	typeName = "polyTransfer"
	apiTypeInt = 849
	apiTypeStr = "kPolyTransfer"
	typeIdInt = 1347700306
	MFnCls = om.MFnDependencyNode
	pass

