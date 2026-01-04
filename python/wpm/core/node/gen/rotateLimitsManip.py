

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	RotateManip = Catalogue.RotateManip
else:
	from .. import retriever
	RotateManip = retriever.getNodeCls("RotateManip")
	assert RotateManip

# add node doc



# region plug type defs

# endregion


# define node class
class RotateLimitsManip(RotateManip):

	# node attributes

	typeName = "rotateLimitsManip"
	apiTypeInt = 217
	apiTypeStr = "kRotateLimitsManip"
	typeIdInt = 1431130700
	MFnCls = om.MFnRotateManip
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

