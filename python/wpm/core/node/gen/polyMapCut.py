

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifier = retriever.getNodeCls("PolyModifier")
assert PolyModifier
if T.TYPE_CHECKING:
	from .. import PolyModifier

# add node doc



# region plug type defs
class MoveRatioPlug(Plug):
	node : PolyMapCut = None
	pass
class UsePinningPlug(Plug):
	node : PolyMapCut = None
	pass
class UvSetNamePlug(Plug):
	node : PolyMapCut = None
	pass
# endregion


# define node class
class PolyMapCut(PolyModifier):
	moveRatio_ : MoveRatioPlug = PlugDescriptor("moveRatio")
	usePinning_ : UsePinningPlug = PlugDescriptor("usePinning")
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")

	# node attributes

	typeName = "polyMapCut"
	apiTypeInt = 413
	apiTypeStr = "kPolyMapCut"
	typeIdInt = 1347240259
	MFnCls = om.MFnDependencyNode
	pass

