

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifierUV = retriever.getNodeCls("PolyModifierUV")
assert PolyModifierUV
if T.TYPE_CHECKING:
	from .. import PolyModifierUV

# add node doc



# region plug type defs
class UvSetNameInputPlug(Plug):
	node : PolyCopyUV = None
	pass
# endregion


# define node class
class PolyCopyUV(PolyModifierUV):
	uvSetNameInput_ : UvSetNameInputPlug = PlugDescriptor("uvSetNameInput")

	# node attributes

	typeName = "polyCopyUV"
	typeIdInt = 1346590038
	pass

