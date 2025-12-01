

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifierWorld = retriever.getNodeCls("PolyModifierWorld")
assert PolyModifierWorld
if T.TYPE_CHECKING:
	from .. import PolyModifierWorld

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

