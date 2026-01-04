

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ConnectionOverride = Catalogue.ConnectionOverride
else:
	from .. import retriever
	ConnectionOverride = retriever.getNodeCls("ConnectionOverride")
	assert ConnectionOverride

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
	nodeLeafClassAttrs = ["targetNodeName"]
	nodeLeafPlugs = ["targetNodeName"]
	pass

