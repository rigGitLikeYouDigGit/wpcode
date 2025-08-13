

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyBlindData = retriever.getNodeCls("PolyBlindData")
assert PolyBlindData
if T.TYPE_CHECKING:
	from .. import PolyBlindData

# add node doc



# region plug type defs

# endregion


# define node class
class SubdBlindData(PolyBlindData):

	# node attributes

	typeName = "subdBlindData"
	apiTypeInt = 802
	apiTypeStr = "kSubdBlindData"
	typeIdInt = 1396851796
	MFnCls = om.MFnDependencyNode
	pass

