

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
class AreaThresholdPlug(Plug):
	node : PolyCollapseF = None
	pass
class UseAreaThresholdPlug(Plug):
	node : PolyCollapseF = None
	pass
# endregion


# define node class
class PolyCollapseF(PolyModifier):
	areaThreshold_ : AreaThresholdPlug = PlugDescriptor("areaThreshold")
	useAreaThreshold_ : UseAreaThresholdPlug = PlugDescriptor("useAreaThreshold")

	# node attributes

	typeName = "polyCollapseF"
	apiTypeInt = 407
	apiTypeStr = "kPolyCollapseF"
	typeIdInt = 1346588486
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["areaThreshold", "useAreaThreshold"]
	nodeLeafPlugs = ["areaThreshold", "useAreaThreshold"]
	pass

