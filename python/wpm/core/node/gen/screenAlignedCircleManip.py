

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
CircleManip = retriever.getNodeCls("CircleManip")
assert CircleManip
if T.TYPE_CHECKING:
	from .. import CircleManip

# add node doc



# region plug type defs

# endregion


# define node class
class ScreenAlignedCircleManip(CircleManip):

	# node attributes

	typeName = "screenAlignedCircleManip"
	apiTypeInt = 127
	apiTypeStr = "kScreenAlignedCircleManip"
	typeIdInt = 1396785997
	MFnCls = om.MFnManip3D
	pass

