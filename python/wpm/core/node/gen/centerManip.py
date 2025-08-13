

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
FreePointManip = retriever.getNodeCls("FreePointManip")
assert FreePointManip
if T.TYPE_CHECKING:
	from .. import FreePointManip

# add node doc



# region plug type defs

# endregion


# define node class
class CenterManip(FreePointManip):

	# node attributes

	typeName = "centerManip"
	apiTypeInt = 134
	apiTypeStr = "kCenterManip"
	typeIdInt = 1129205072
	MFnCls = om.MFnManip3D
	pass

