

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
CurveFromSubdiv = retriever.getNodeCls("CurveFromSubdiv")
assert CurveFromSubdiv
if T.TYPE_CHECKING:
	from .. import CurveFromSubdiv

# add node doc



# region plug type defs
class EdgeIndexLPlug(Plug):
	node : CurveFromSubdivEdge = None
	pass
class EdgeIndexRPlug(Plug):
	node : CurveFromSubdivEdge = None
	pass
# endregion


# define node class
class CurveFromSubdivEdge(CurveFromSubdiv):
	edgeIndexL_ : EdgeIndexLPlug = PlugDescriptor("edgeIndexL")
	edgeIndexR_ : EdgeIndexRPlug = PlugDescriptor("edgeIndexR")

	# node attributes

	typeName = "curveFromSubdivEdge"
	apiTypeInt = 836
	apiTypeStr = "kCurveFromSubdivEdge"
	typeIdInt = 1396921157
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["edgeIndexL", "edgeIndexR"]
	nodeLeafPlugs = ["edgeIndexL", "edgeIndexR"]
	pass

