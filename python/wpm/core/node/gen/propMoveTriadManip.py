

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
FreePointTriadManip = retriever.getNodeCls("FreePointTriadManip")
assert FreePointTriadManip
if T.TYPE_CHECKING:
	from .. import FreePointTriadManip

# add node doc



# region plug type defs

# endregion


# define node class
class PropMoveTriadManip(FreePointTriadManip):

	# node attributes

	typeName = "propMoveTriadManip"
	apiTypeInt = 138
	apiTypeStr = "kPropMoveTriadManip"
	typeIdInt = 1431130196
	MFnCls = om.MFnFreePointTriadManip
	pass

