

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
class QuadPtOnLineManip(ManipContainer):

	# node attributes

	typeName = "quadPtOnLineManip"
	apiTypeInt = 179
	apiTypeStr = "kQuadPtOnLineManip"
	typeIdInt = 1431392332
	MFnCls = om.MFnManip3D
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

