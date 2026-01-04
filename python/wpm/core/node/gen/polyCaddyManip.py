

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	CaddyManipBase = Catalogue.CaddyManipBase
else:
	from .. import retriever
	CaddyManipBase = retriever.getNodeCls("CaddyManipBase")
	assert CaddyManipBase

# add node doc



# region plug type defs

# endregion


# define node class
class PolyCaddyManip(CaddyManipBase):

	# node attributes

	typeName = "polyCaddyManip"
	apiTypeInt = 1111
	apiTypeStr = "kPolyCaddyManip"
	typeIdInt = 1346585677
	MFnCls = om.MFnManip3D
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

