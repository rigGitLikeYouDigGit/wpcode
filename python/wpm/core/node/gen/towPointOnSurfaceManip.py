

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PointOnSurfManip = retriever.getNodeCls("PointOnSurfManip")
assert PointOnSurfManip
if T.TYPE_CHECKING:
	from .. import PointOnSurfManip

# add node doc



# region plug type defs

# endregion


# define node class
class TowPointOnSurfaceManip(PointOnSurfManip):

	# node attributes

	typeName = "towPointOnSurfaceManip"
	apiTypeInt = 776
	apiTypeStr = "kTowPointOnSurfaceManip"
	typeIdInt = 1431588947
	MFnCls = om.MFnPointOnSurfaceManip
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

