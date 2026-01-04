

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	SimpleSelector = Catalogue.SimpleSelector
else:
	from .. import retriever
	SimpleSelector = retriever.getNodeCls("SimpleSelector")
	assert SimpleSelector

# add node doc



# region plug type defs

# endregion


# define node class
class LightsCollectionSelector(SimpleSelector):

	# node attributes

	typeName = "lightsCollectionSelector"
	typeIdInt = 1476395940
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

