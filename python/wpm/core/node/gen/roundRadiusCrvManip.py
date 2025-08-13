

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
RoundRadiusManip = retriever.getNodeCls("RoundRadiusManip")
assert RoundRadiusManip
if T.TYPE_CHECKING:
	from .. import RoundRadiusManip

# add node doc



# region plug type defs

# endregion


# define node class
class RoundRadiusCrvManip(RoundRadiusManip):

	# node attributes

	typeName = "roundRadiusCrvManip"
	apiTypeInt = 647
	apiTypeStr = "kRoundRadiusCrvManip"
	typeIdInt = 1431130703
	MFnCls = om.MFnManip3D
	pass

