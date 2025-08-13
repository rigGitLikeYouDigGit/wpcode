

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
class IkPASolver(IkSolver):

	# node attributes

	typeName = "ikPASolver"
	typeIdInt = 1263550803
	pass

