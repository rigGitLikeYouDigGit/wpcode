

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ControlPoint = Catalogue.ControlPoint
else:
	from .. import retriever
	ControlPoint = retriever.getNodeCls("ControlPoint")
	assert ControlPoint

# add node doc



# region plug type defs

# endregion


# define node class
class CurveShape(ControlPoint):

	# node attributes

	typeName = "curveShape"
	typeIdInt = 1313035859
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

