

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
CurveFromMesh = retriever.getNodeCls("CurveFromMesh")
assert CurveFromMesh
if T.TYPE_CHECKING:
	from .. import CurveFromMesh

# add node doc



# region plug type defs
class CurveOnMeshPlug(Plug):
	node : CurveFromMeshCoM = None
	pass
# endregion


# define node class
class CurveFromMeshCoM(CurveFromMesh):
	curveOnMesh_ : CurveOnMeshPlug = PlugDescriptor("curveOnMesh")

	# node attributes

	typeName = "curveFromMeshCoM"
	apiTypeInt = 934
	apiTypeStr = "kCurveFromMeshCoM"
	typeIdInt = 1313033539
	MFnCls = om.MFnDependencyNode
	pass

