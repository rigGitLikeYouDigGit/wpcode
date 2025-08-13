

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
RevolvedPrimitive = retriever.getNodeCls("RevolvedPrimitive")
assert RevolvedPrimitive
if T.TYPE_CHECKING:
	from .. import RevolvedPrimitive

# add node doc



# region plug type defs
class MinorSweepPlug(Plug):
	node : MakeNurbTorus = None
	pass
# endregion


# define node class
class MakeNurbTorus(RevolvedPrimitive):
	minorSweep_ : MinorSweepPlug = PlugDescriptor("minorSweep")

	# node attributes

	typeName = "makeNurbTorus"
	typeIdInt = 1314148178
	pass

