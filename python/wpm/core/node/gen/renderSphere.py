

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ImplicitSphere = retriever.getNodeCls("ImplicitSphere")
assert ImplicitSphere
if T.TYPE_CHECKING:
	from .. import ImplicitSphere

# add node doc



# region plug type defs

# endregion


# define node class
class RenderSphere(ImplicitSphere):

	# node attributes

	typeName = "renderSphere"
	apiTypeInt = 298
	apiTypeStr = "kRenderSphere"
	typeIdInt = 1380864848
	MFnCls = om.MFnDagNode
	pass

