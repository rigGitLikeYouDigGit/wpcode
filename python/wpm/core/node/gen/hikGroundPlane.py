

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Transform = retriever.getNodeCls("Transform")
assert Transform
if T.TYPE_CHECKING:
	from .. import Transform

# add node doc



# region plug type defs
class LengthPlug(Plug):
	node : HikGroundPlane = None
	pass
# endregion


# define node class
class HikGroundPlane(Transform):
	length_ : LengthPlug = PlugDescriptor("length")

	# node attributes

	typeName = "hikGroundPlane"
	apiTypeInt = 984
	apiTypeStr = "kHikGroundPlane"
	typeIdInt = 1212632644
	MFnCls = om.MFnTransform
	pass

