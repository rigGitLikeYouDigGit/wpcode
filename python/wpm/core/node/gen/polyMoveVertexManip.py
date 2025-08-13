

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ManipContainer = retriever.getNodeCls("ManipContainer")
assert ManipContainer
if T.TYPE_CHECKING:
	from .. import ManipContainer

# add node doc



# region plug type defs

# endregion


# define node class
class PolyMoveVertexManip(ManipContainer):

	# node attributes

	typeName = "polyMoveVertexManip"
	apiTypeInt = 196
	apiTypeStr = "kPolyMoveVertexManip"
	typeIdInt = 1431328333
	MFnCls = om.MFnDagNode
	pass

