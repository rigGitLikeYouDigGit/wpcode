

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
	node : SurfaceVarGroup = None
	pass
class LocalPlug(Plug):
	node : SurfaceVarGroup = None
	pass
# endregion


# define node class
class SurfaceVarGroup(BaseGeometryVarGroup):
	create_ : CreatePlug = PlugDescriptor("create")
	local_ : LocalPlug = PlugDescriptor("local")

	# node attributes

	typeName = "surfaceVarGroup"
	apiTypeInt = 118
	apiTypeStr = "kSurfaceVarGroup"
	typeIdInt = 1314084423
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["create", "local"]
	nodeLeafPlugs = ["create", "local"]
	pass

