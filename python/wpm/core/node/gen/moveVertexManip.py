

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Manip3D = retriever.getNodeCls("Manip3D")
assert Manip3D
if T.TYPE_CHECKING:
	from .. import Manip3D

# add node doc



# region plug type defs

# endregion


# define node class
class MoveVertexManip(Manip3D):

	# node attributes

	typeName = "moveVertexManip"
	apiTypeInt = 763
	apiTypeStr = "kMoveVertexManip"
	typeIdInt = 1431721037
	MFnCls = om.MFnManip3D
	pass

