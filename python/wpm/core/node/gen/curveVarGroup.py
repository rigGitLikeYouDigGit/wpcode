

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
BaseGeometryVarGroup = retriever.getNodeCls("BaseGeometryVarGroup")
assert BaseGeometryVarGroup
if T.TYPE_CHECKING:
	from .. import BaseGeometryVarGroup

# add node doc



# region plug type defs
class CreatePlug(Plug):
	node : CurveVarGroup = None
	pass
class DisplaySmoothnessPlug(Plug):
	node : CurveVarGroup = None
	pass
class LocalPlug(Plug):
	node : CurveVarGroup = None
	pass
# endregion


# define node class
class CurveVarGroup(BaseGeometryVarGroup):
	create_ : CreatePlug = PlugDescriptor("create")
	displaySmoothness_ : DisplaySmoothnessPlug = PlugDescriptor("displaySmoothness")
	local_ : LocalPlug = PlugDescriptor("local")

	# node attributes

	typeName = "curveVarGroup"
	apiTypeInt = 116
	apiTypeStr = "kCurveVarGroup"
	typeIdInt = 1313035847
	MFnCls = om.MFnTransform
	pass

