

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
class AlphaPlug(Plug):
	node : PolyAverageVertex = None
	pass
class BetaPlug(Plug):
	node : PolyAverageVertex = None
	pass
class IterationsPlug(Plug):
	node : PolyAverageVertex = None
	pass
# endregion


# define node class
class PolyAverageVertex(PolyModifierWorld):
	alpha_ : AlphaPlug = PlugDescriptor("alpha")
	beta_ : BetaPlug = PlugDescriptor("beta")
	iterations_ : IterationsPlug = PlugDescriptor("iterations")

	# node attributes

	typeName = "polyAverageVertex"
	apiTypeInt = 850
	apiTypeStr = "kPolyAverageVertex"
	typeIdInt = 1346459222
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["alpha", "beta", "iterations"]
	nodeLeafPlugs = ["alpha", "beta", "iterations"]
	pass

