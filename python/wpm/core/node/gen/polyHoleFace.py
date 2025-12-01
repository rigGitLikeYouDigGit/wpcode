

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
class PolyHoleFace(PolyModifierWorld):

	# node attributes

	typeName = "polyHoleFace"
	apiTypeInt = 1059
	apiTypeStr = "kPolyHoleFace"
	typeIdInt = 1346913861
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

