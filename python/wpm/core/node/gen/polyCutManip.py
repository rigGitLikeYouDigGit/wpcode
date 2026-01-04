

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ProjectionManip = Catalogue.ProjectionManip
else:
	from .. import retriever
	ProjectionManip = retriever.getNodeCls("ProjectionManip")
	assert ProjectionManip

# add node doc



# region plug type defs

# endregion


# define node class
class PolyCutManip(ProjectionManip):

	# node attributes

	typeName = "polyCutManip"
	apiTypeInt = 905
	apiTypeStr = "kPolyCutManip"
	typeIdInt = 1346587985
	MFnCls = om.MFnManip3D
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

