

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifierWorld = Catalogue.PolyModifierWorld
else:
	from .. import retriever
	PolyModifierWorld = retriever.getNodeCls("PolyModifierWorld")
	assert PolyModifierWorld

# add node doc



# region plug type defs

# endregion


# define node class
class PolyPassThru(PolyModifierWorld):

	# node attributes

	typeName = "polyPassThru"
	apiTypeInt = 1122
	apiTypeStr = "kPolyPassThru"
	typeIdInt = 1348030548
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

