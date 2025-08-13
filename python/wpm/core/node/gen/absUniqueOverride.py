

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AbsOverride = retriever.getNodeCls("AbsOverride")
assert AbsOverride
if T.TYPE_CHECKING:
	from .. import AbsOverride

# add node doc



# region plug type defs
class TargetNodeNamePlug(Plug):
	node : AbsUniqueOverride = None
	pass
# endregion


# define node class
class AbsUniqueOverride(AbsOverride):
	targetNodeName_ : TargetNodeNamePlug = PlugDescriptor("targetNodeName")

	# node attributes

	typeName = "absUniqueOverride"
	typeIdInt = 1476395936
	pass

