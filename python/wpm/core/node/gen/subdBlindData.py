

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyBlindData = Catalogue.PolyBlindData
else:
	from .. import retriever
	PolyBlindData = retriever.getNodeCls("PolyBlindData")
	assert PolyBlindData

# add node doc



# region plug type defs

# endregion


# define node class
class SubdBlindData(PolyBlindData):

	# node attributes

	typeName = "subdBlindData"
	apiTypeInt = 802
	apiTypeStr = "kSubdBlindData"
	typeIdInt = 1396851796
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

