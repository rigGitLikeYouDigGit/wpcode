

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifierManip = retriever.getNodeCls("PolyModifierManip")
assert PolyModifierManip
if T.TYPE_CHECKING:
	from .. import PolyModifierManip

# add node doc



# region plug type defs

# endregion


# define node class
class PolyModifierManipContainer(PolyModifierManip):

	# node attributes

	typeName = "polyModifierManipContainer"
	apiTypeInt = 1112
	apiTypeStr = "kPolyModifierManipContainer"
	typeIdInt = 1347243331
	MFnCls = om.MFnDagNode
	pass

