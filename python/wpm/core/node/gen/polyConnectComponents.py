

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
class AdjustEdgeFlowPlug(Plug):
	node : PolyConnectComponents = None
	pass
class InsertWithEdgeFlowPlug(Plug):
	node : PolyConnectComponents = None
	pass
# endregion


# define node class
class PolyConnectComponents(PolyModifier):
	adjustEdgeFlow_ : AdjustEdgeFlowPlug = PlugDescriptor("adjustEdgeFlow")
	insertWithEdgeFlow_ : InsertWithEdgeFlowPlug = PlugDescriptor("insertWithEdgeFlow")

	# node attributes

	typeName = "polyConnectComponents"
	apiTypeInt = 1061
	apiTypeStr = "kPolyConnectComponents"
	typeIdInt = 1346585427
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["adjustEdgeFlow", "insertWithEdgeFlow"]
	nodeLeafPlugs = ["adjustEdgeFlow", "insertWithEdgeFlow"]
	pass

