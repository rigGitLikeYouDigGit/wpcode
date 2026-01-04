

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	CurveFromMesh = Catalogue.CurveFromMesh
else:
	from .. import retriever
	CurveFromMesh = retriever.getNodeCls("CurveFromMesh")
	assert CurveFromMesh

# add node doc



# region plug type defs
class EdgeIndexPlug(Plug):
	node : CurveFromMeshEdge = None
	pass
# endregion


# define node class
class CurveFromMeshEdge(CurveFromMesh):
	edgeIndex_ : EdgeIndexPlug = PlugDescriptor("edgeIndex")

	# node attributes

	typeName = "curveFromMeshEdge"
	apiTypeInt = 640
	apiTypeStr = "kCurveFromMeshEdge"
	typeIdInt = 1313033541
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["edgeIndex"]
	nodeLeafPlugs = ["edgeIndex"]
	pass

