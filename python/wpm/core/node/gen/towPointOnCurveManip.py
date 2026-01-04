

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PointOnCurveManip = Catalogue.PointOnCurveManip
else:
	from .. import retriever
	PointOnCurveManip = retriever.getNodeCls("PointOnCurveManip")
	assert PointOnCurveManip

# add node doc



# region plug type defs

# endregion


# define node class
class TowPointOnCurveManip(PointOnCurveManip):

	# node attributes

	typeName = "towPointOnCurveManip"
	apiTypeInt = 209
	apiTypeStr = "kTowPointOnCurveManip"
	typeIdInt = 1431588931
	MFnCls = om.MFnPointOnCurveManip
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

