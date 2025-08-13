

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ShadingDependNode = retriever.getNodeCls("ShadingDependNode")
assert ShadingDependNode
if T.TYPE_CHECKING:
	from .. import ShadingDependNode

# add node doc



# region plug type defs

# endregion


# define node class
class PaintableShadingDependNode(ShadingDependNode):

	# node attributes

	typeName = "paintableShadingDependNode"
	typeIdInt = 1347634254
	pass

