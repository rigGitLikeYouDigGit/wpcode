

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
class UvSetNamePlug(Plug):
	node : PolyMapDel = None
	pass
# endregion


# define node class
class PolyMapDel(PolyModifier):
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")

	# node attributes

	typeName = "polyMapDel"
	apiTypeInt = 414
	apiTypeStr = "kPolyMapDel"
	typeIdInt = 1347240260
	MFnCls = om.MFnDependencyNode
	pass

