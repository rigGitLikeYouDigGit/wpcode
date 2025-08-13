

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
class SmoothnessPlug(Plug):
	node : PolySmooth = None
	pass
# endregion


# define node class
class PolySmooth(PolyModifier):
	smoothness_ : SmoothnessPlug = PlugDescriptor("smoothness")

	# node attributes

	typeName = "polySmooth"
	apiTypeInt = 428
	apiTypeStr = "kPolySmooth"
	typeIdInt = 1347636564
	MFnCls = om.MFnDependencyNode
	pass

