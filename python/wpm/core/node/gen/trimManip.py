

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PointOnSurfaceManip = Catalogue.PointOnSurfaceManip
else:
	from .. import retriever
	PointOnSurfaceManip = retriever.getNodeCls("PointOnSurfaceManip")
	assert PointOnSurfaceManip

# add node doc



# region plug type defs

# endregion


# define node class
class TrimManip(PointOnSurfaceManip):

	# node attributes

	typeName = "trimManip"
	apiTypeInt = 228
	apiTypeStr = "kTrimManip"
	typeIdInt = 1431131213
	MFnCls = om.MFnPointOnSurfaceManip
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

