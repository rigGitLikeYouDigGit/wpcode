

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
class IkSCsolver(IkSolver):

	# node attributes

	typeName = "ikSCsolver"
	typeIdInt = 1263747923
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

