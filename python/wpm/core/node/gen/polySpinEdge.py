

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
class OffsetPlug(Plug):
	node : PolySpinEdge = None
	pass
class ReversePlug(Plug):
	node : PolySpinEdge = None
	pass
# endregion


# define node class
class PolySpinEdge(PolyModifier):
	offset_ : OffsetPlug = PlugDescriptor("offset")
	reverse_ : ReversePlug = PlugDescriptor("reverse")

	# node attributes

	typeName = "polySpinEdge"
	apiTypeInt = 1058
	apiTypeStr = "kPolySpinEdge"
	typeIdInt = 1347637329
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["offset", "reverse"]
	nodeLeafPlugs = ["offset", "reverse"]
	pass

