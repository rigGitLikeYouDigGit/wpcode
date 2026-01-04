

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	AnimBlend = Catalogue.AnimBlend
else:
	from .. import retriever
	AnimBlend = retriever.getNodeCls("AnimBlend")
	assert AnimBlend

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
	nodeLeafClassAttrs = ["rotateInterp"]
	nodeLeafPlugs = ["rotateInterp"]
	pass

