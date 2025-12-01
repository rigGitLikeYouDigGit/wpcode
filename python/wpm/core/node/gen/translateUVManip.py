

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PointOnSurfaceManip = retriever.getNodeCls("PointOnSurfaceManip")
assert PointOnSurfaceManip
if T.TYPE_CHECKING:
	from .. import PointOnSurfaceManip

# add node doc



# region plug type defs

# endregion


# define node class
class TranslateUVManip(PointOnSurfaceManip):

	# node attributes

	typeName = "translateUVManip"
	apiTypeInt = 213
	apiTypeStr = "kTranslateUVManip"
	typeIdInt = 1431131478
	MFnCls = om.MFnPointOnSurfaceManip
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

