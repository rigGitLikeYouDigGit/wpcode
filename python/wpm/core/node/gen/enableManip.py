

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
LimitManip = retriever.getNodeCls("LimitManip")
assert LimitManip
if T.TYPE_CHECKING:
	from .. import LimitManip

# add node doc



# region plug type defs

# endregion


# define node class
class EnableManip(LimitManip):

	# node attributes

	typeName = "enableManip"
	apiTypeInt = 136
	apiTypeStr = "kEnableManip"
	typeIdInt = 1162759504
	MFnCls = om.MFnManip3D
	pass

