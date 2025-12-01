

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
class UsePinningPlug(Plug):
	node : PolyMapSew = None
	pass
class UvSetNamePlug(Plug):
	node : PolyMapSew = None
	pass
# endregion


# define node class
class PolyMapSew(PolyModifier):
	usePinning_ : UsePinningPlug = PlugDescriptor("usePinning")
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")

	# node attributes

	typeName = "polyMapSew"
	apiTypeInt = 415
	apiTypeStr = "kPolyMapSew"
	typeIdInt = 1347240275
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["usePinning", "uvSetName"]
	nodeLeafPlugs = ["usePinning", "uvSetName"]
	pass

