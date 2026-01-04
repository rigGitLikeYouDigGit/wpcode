

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	RevolvedPrimitive = Catalogue.RevolvedPrimitive
else:
	from .. import retriever
	RevolvedPrimitive = retriever.getNodeCls("RevolvedPrimitive")
	assert RevolvedPrimitive

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
	nodeLeafClassAttrs = ["minorSweep"]
	nodeLeafPlugs = ["minorSweep"]
	pass

