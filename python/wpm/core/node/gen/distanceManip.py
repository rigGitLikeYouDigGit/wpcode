

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PointOnLineManip = retriever.getNodeCls("PointOnLineManip")
assert PointOnLineManip
if T.TYPE_CHECKING:
	from .. import PointOnLineManip

# add node doc



# region plug type defs

# endregion


# define node class
class DistanceManip(PointOnLineManip):

	# node attributes

	typeName = "distanceManip"
	apiTypeInt = 638
	apiTypeStr = "kDistanceManip"
	typeIdInt = 1431127117
	MFnCls = om.MFnDistanceManip
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

