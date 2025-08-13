

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifierUV = retriever.getNodeCls("PolyModifierUV")
assert PolyModifierUV
if T.TYPE_CHECKING:
	from .. import PolyModifierUV

# add node doc



# region plug type defs
class DistancePlug(Plug):
	node : PolyMergeUV = None
	pass
# endregion


# define node class
class PolyMergeUV(PolyModifierUV):
	distance_ : DistancePlug = PlugDescriptor("distance")

	# node attributes

	typeName = "polyMergeUV"
	apiTypeInt = 910
	apiTypeStr = "kPolyMergeUV"
	typeIdInt = 1347241813
	MFnCls = om.MFnDependencyNode
	pass

