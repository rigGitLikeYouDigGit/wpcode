

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ManipContainer = retriever.getNodeCls("ManipContainer")
assert ManipContainer
if T.TYPE_CHECKING:
	from .. import ManipContainer

# add node doc



# region plug type defs

# endregion


# define node class
class RevolveManip(ManipContainer):

	# node attributes

	typeName = "revolveManip"
	apiTypeInt = 184
	apiTypeStr = "kRevolveManip"
	typeIdInt = 1431130710
	MFnCls = om.MFnManip3D
	pass

