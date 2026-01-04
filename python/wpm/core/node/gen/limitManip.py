

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	CenterManip = Catalogue.CenterManip
else:
	from .. import retriever
	CenterManip = retriever.getNodeCls("CenterManip")
	assert CenterManip

# add node doc



# region plug type defs

# endregion


# define node class
class LimitManip(CenterManip):

	# node attributes

	typeName = "limitManip"
	apiTypeInt = 135
	apiTypeStr = "kLimitManip"
	typeIdInt = 1280593232
	MFnCls = om.MFnManip3D
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

