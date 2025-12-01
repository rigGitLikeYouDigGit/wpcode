

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PointConstraint = retriever.getNodeCls("PointConstraint")
assert PointConstraint
if T.TYPE_CHECKING:
	from .. import PointConstraint

# add node doc



# region plug type defs
class PivotSpacePlug(Plug):
	node : PoleVectorConstraint = None
	pass
# endregion


# define node class
class PoleVectorConstraint(PointConstraint):
	pivotSpace_ : PivotSpacePlug = PlugDescriptor("pivotSpace")

	# node attributes

	typeName = "poleVectorConstraint"
	apiTypeInt = 243
	apiTypeStr = "kPoleVectorConstraint"
	typeIdInt = 1146115651
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["pivotSpace"]
	nodeLeafPlugs = ["pivotSpace"]
	pass

