

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ImplicitBox = retriever.getNodeCls("ImplicitBox")
assert ImplicitBox
if T.TYPE_CHECKING:
	from .. import ImplicitBox

# add node doc



# region plug type defs

# endregion


# define node class
class RenderBox(ImplicitBox):

	# node attributes

	typeName = "renderBox"
	apiTypeInt = 868
	apiTypeStr = "kRenderBox"
	typeIdInt = 1380860504
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

