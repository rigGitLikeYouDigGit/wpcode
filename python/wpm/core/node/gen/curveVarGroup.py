

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	BaseGeometryVarGroup = Catalogue.BaseGeometryVarGroup
else:
	from .. import retriever
	BaseGeometryVarGroup = retriever.getNodeCls("BaseGeometryVarGroup")
	assert BaseGeometryVarGroup

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
	nodeLeafClassAttrs = ["create", "displaySmoothness", "local"]
	nodeLeafPlugs = ["create", "displaySmoothness", "local"]
	pass

