

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ControlPoint = retriever.getNodeCls("ControlPoint")
assert ControlPoint
if T.TYPE_CHECKING:
	from .. import ControlPoint

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

