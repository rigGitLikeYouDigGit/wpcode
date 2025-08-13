

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AnimBlend = retriever.getNodeCls("AnimBlend")
assert AnimBlend
if T.TYPE_CHECKING:
	from .. import AnimBlend

# add node doc



# region plug type defs
class RotateInterpPlug(Plug):
	node : AnimBlendInOut = None
	pass
# endregion


# define node class
class AnimBlendInOut(AnimBlend):
	rotateInterp_ : RotateInterpPlug = PlugDescriptor("rotateInterp")

	# node attributes

	typeName = "animBlendInOut"
	apiTypeInt = 795
	apiTypeStr = "kAnimBlendInOut"
	typeIdInt = 1094863183
	MFnCls = om.MFnDependencyNode
	pass

