

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyToolFeedbackManip = Catalogue.PolyToolFeedbackManip
else:
	from .. import retriever
	PolyToolFeedbackManip = retriever.getNodeCls("PolyToolFeedbackManip")
	assert PolyToolFeedbackManip

# add node doc



# region plug type defs

# endregion


# define node class
class PolySelectEditFeedbackManip(PolyToolFeedbackManip):

	# node attributes

	typeName = "polySelectEditFeedbackManip"
	apiTypeInt = 1042
	apiTypeStr = "kPolySelectEditFeedbackManip"
	typeIdInt = 1397048909
	MFnCls = om.MFnManip3D
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

