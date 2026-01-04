

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	DeformFunc = Catalogue.DeformFunc
else:
	from .. import retriever
	DeformFunc = retriever.getNodeCls("DeformFunc")
	assert DeformFunc

# add node doc



# region plug type defs
class EndAnglePlug(Plug):
	node : DeformTwist = None
	pass
class HighBoundPlug(Plug):
	node : DeformTwist = None
	pass
class LowBoundPlug(Plug):
	node : DeformTwist = None
	pass
class StartAnglePlug(Plug):
	node : DeformTwist = None
	pass
# endregion


# define node class
class DeformTwist(DeformFunc):
	endAngle_ : EndAnglePlug = PlugDescriptor("endAngle")
	highBound_ : HighBoundPlug = PlugDescriptor("highBound")
	lowBound_ : LowBoundPlug = PlugDescriptor("lowBound")
	startAngle_ : StartAnglePlug = PlugDescriptor("startAngle")

	# node attributes

	typeName = "deformTwist"
	apiTypeInt = 626
	apiTypeStr = "kDeformTwist"
	typeIdInt = 1178883159
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["endAngle", "highBound", "lowBound", "startAngle"]
	nodeLeafPlugs = ["endAngle", "highBound", "lowBound", "startAngle"]
	pass

