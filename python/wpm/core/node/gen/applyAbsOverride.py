

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ApplyOverride = retriever.getNodeCls("ApplyOverride")
assert ApplyOverride
if T.TYPE_CHECKING:
	from .. import ApplyOverride

# add node doc



# region plug type defs

# endregion


# define node class
class ApplyAbsOverride(ApplyOverride):

	# node attributes

	typeName = "applyAbsOverride"
	typeIdInt = 1476395897
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

