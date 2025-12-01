

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
class EndSmoothnessPlug(Plug):
	node : DeformSquash = None
	pass
class ExpandPlug(Plug):
	node : DeformSquash = None
	pass
class FactorPlug(Plug):
	node : DeformSquash = None
	pass
class HighBoundPlug(Plug):
	node : DeformSquash = None
	pass
class LowBoundPlug(Plug):
	node : DeformSquash = None
	pass
class MaxExpandPosPlug(Plug):
	node : DeformSquash = None
	pass
class StartSmoothnessPlug(Plug):
	node : DeformSquash = None
	pass
# endregion


# define node class
class DeformSquash(DeformFunc):
	endSmoothness_ : EndSmoothnessPlug = PlugDescriptor("endSmoothness")
	expand_ : ExpandPlug = PlugDescriptor("expand")
	factor_ : FactorPlug = PlugDescriptor("factor")
	highBound_ : HighBoundPlug = PlugDescriptor("highBound")
	lowBound_ : LowBoundPlug = PlugDescriptor("lowBound")
	maxExpandPos_ : MaxExpandPosPlug = PlugDescriptor("maxExpandPos")
	startSmoothness_ : StartSmoothnessPlug = PlugDescriptor("startSmoothness")

	# node attributes

	typeName = "deformSquash"
	apiTypeInt = 627
	apiTypeStr = "kDeformSquash"
	typeIdInt = 1178882897
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["endSmoothness", "expand", "factor", "highBound", "lowBound", "maxExpandPos", "startSmoothness"]
	nodeLeafPlugs = ["endSmoothness", "expand", "factor", "highBound", "lowBound", "maxExpandPos", "startSmoothness"]
	pass

