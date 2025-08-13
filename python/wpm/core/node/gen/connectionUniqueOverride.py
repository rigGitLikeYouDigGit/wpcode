

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ConnectionOverride = retriever.getNodeCls("ConnectionOverride")
assert ConnectionOverride
if T.TYPE_CHECKING:
	from .. import ConnectionOverride

# add node doc



# region plug type defs
class TargetNodeNamePlug(Plug):
	node : ConnectionUniqueOverride = None
	pass
# endregion


# define node class
class ConnectionUniqueOverride(ConnectionOverride):
	targetNodeName_ : TargetNodeNamePlug = PlugDescriptor("targetNodeName")

	# node attributes

	typeName = "connectionUniqueOverride"
	typeIdInt = 1476395938
	pass

