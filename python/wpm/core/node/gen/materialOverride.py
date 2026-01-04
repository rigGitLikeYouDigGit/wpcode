

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

# endregion


# define node class
class MaterialOverride(ConnectionOverride):

	# node attributes

	typeName = "materialOverride"
	typeIdInt = 1476395911
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

