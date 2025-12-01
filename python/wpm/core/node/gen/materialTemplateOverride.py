

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

# endregion


# define node class
class MaterialTemplateOverride(ConnectionOverride):

	# node attributes

	typeName = "materialTemplateOverride"
	typeIdInt = 1476395913
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

