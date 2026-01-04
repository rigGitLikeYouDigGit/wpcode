

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	LightGroup = Catalogue.LightGroup
else:
	from .. import retriever
	LightGroup = retriever.getNodeCls("LightGroup")
	assert LightGroup

# add node doc



# region plug type defs

# endregion


# define node class
class LightEditor(LightGroup):

	# node attributes

	typeName = "lightEditor"
	typeIdInt = 1476396003
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

