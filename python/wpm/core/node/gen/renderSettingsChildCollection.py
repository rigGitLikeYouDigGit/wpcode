

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Collection = Catalogue.Collection
else:
	from .. import retriever
	Collection = retriever.getNodeCls("Collection")
	assert Collection

# add node doc



# region plug type defs

# endregion


# define node class
class RenderSettingsChildCollection(Collection):

	# node attributes

	typeName = "renderSettingsChildCollection"
	typeIdInt = 1476395939
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

