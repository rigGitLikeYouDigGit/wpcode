

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	IkSolver = Catalogue.IkSolver
else:
	from .. import retriever
	IkSolver = retriever.getNodeCls("IkSolver")
	assert IkSolver

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
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

