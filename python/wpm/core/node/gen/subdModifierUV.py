

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	SubdModifierWorld = Catalogue.SubdModifierWorld
else:
	from .. import retriever
	SubdModifierWorld = retriever.getNodeCls("SubdModifierWorld")
	assert SubdModifierWorld

# add node doc



# region plug type defs

# endregion


# define node class
class SubdModifierUV(SubdModifierWorld):

	# node attributes

	typeName = "subdModifierUV"
	typeIdInt = 1397577078
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

