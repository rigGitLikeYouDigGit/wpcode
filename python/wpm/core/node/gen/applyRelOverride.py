

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ApplyOverride = Catalogue.ApplyOverride
else:
	from .. import retriever
	ApplyOverride = retriever.getNodeCls("ApplyOverride")
	assert ApplyOverride

# add node doc



# region plug type defs

# endregion


# define node class
class ApplyRelOverride(ApplyOverride):

	# node attributes

	typeName = "applyRelOverride"
	typeIdInt = 1476395899
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

