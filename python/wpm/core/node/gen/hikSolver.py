

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
IkSolver = retriever.getNodeCls("IkSolver")
assert IkSolver
if T.TYPE_CHECKING:
	from .. import IkSolver

# add node doc



# region plug type defs

# endregion


# define node class
class HikSolver(IkSolver):

	# node attributes

	typeName = "hikSolver"
	apiTypeInt = 964
	apiTypeStr = "kHikSolver"
	typeIdInt = 1263028555
	MFnCls = om.MFnDependencyNode
	pass

