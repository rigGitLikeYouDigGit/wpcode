

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	LimitManip = Catalogue.LimitManip
else:
	from .. import retriever
	LimitManip = retriever.getNodeCls("LimitManip")
	assert LimitManip

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
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

