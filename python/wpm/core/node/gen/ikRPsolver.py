

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
IkSCsolver = retriever.getNodeCls("IkSCsolver")
assert IkSCsolver
if T.TYPE_CHECKING:
	from .. import IkSCsolver

# add node doc



# region plug type defs

# endregion


# define node class
class IkRPsolver(IkSCsolver):

	# node attributes

	typeName = "ikRPsolver"
	typeIdInt = 1263685715
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

