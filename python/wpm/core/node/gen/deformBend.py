

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
DeformFunc = retriever.getNodeCls("DeformFunc")
assert DeformFunc
if T.TYPE_CHECKING:
	from .. import DeformFunc

# add node doc



# region plug type defs
class CurvaturePlug(Plug):
	node : DeformBend = None
	pass
class HighBoundPlug(Plug):
	node : DeformBend = None
	pass
class LowBoundPlug(Plug):
	node : DeformBend = None
	pass
# endregion


# define node class
class DeformBend(DeformFunc):
	curvature_ : CurvaturePlug = PlugDescriptor("curvature")
	highBound_ : HighBoundPlug = PlugDescriptor("highBound")
	lowBound_ : LowBoundPlug = PlugDescriptor("lowBound")

	# node attributes

	typeName = "deformBend"
	apiTypeInt = 625
	apiTypeStr = "kDeformBend"
	typeIdInt = 1178878532
	MFnCls = om.MFnDagNode
	pass

