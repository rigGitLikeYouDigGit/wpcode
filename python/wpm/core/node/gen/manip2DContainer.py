

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Manip2D = retriever.getNodeCls("Manip2D")
assert Manip2D
if T.TYPE_CHECKING:
	from .. import Manip2D

# add node doc



# region plug type defs

# endregion


# define node class
class Manip2DContainer(Manip2D):

	# node attributes

	typeName = "manip2DContainer"
	apiTypeInt = 192
	apiTypeStr = "kManip2DContainer"
	typeIdInt = 1431122499
	MFnCls = om.MFnTransform
	pass

