

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ImplicitCone = retriever.getNodeCls("ImplicitCone")
assert ImplicitCone
if T.TYPE_CHECKING:
	from .. import ImplicitCone

# add node doc



# region plug type defs

# endregion


# define node class
class RenderCone(ImplicitCone):

	# node attributes

	typeName = "renderCone"
	apiTypeInt = 97
	apiTypeStr = "kRenderCone"
	typeIdInt = 1380860751
	MFnCls = om.MFnDagNode
	pass

