

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
class RbfSrfManip(ManipContainer):

	# node attributes

	typeName = "rbfSrfManip"
	apiTypeInt = 180
	apiTypeStr = "kRbfSrfManip"
	typeIdInt = 1431127635
	MFnCls = om.MFnManip3D
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

