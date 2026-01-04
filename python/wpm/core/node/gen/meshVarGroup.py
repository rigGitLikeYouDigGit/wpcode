

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
	node : MeshVarGroup = None
	pass
class LocalPlug(Plug):
	node : MeshVarGroup = None
	pass
# endregion


# define node class
class MeshVarGroup(BaseGeometryVarGroup):
	create_ : CreatePlug = PlugDescriptor("create")
	local_ : LocalPlug = PlugDescriptor("local")

	# node attributes

	typeName = "meshVarGroup"
	apiTypeInt = 117
	apiTypeStr = "kMeshVarGroup"
	typeIdInt = 1313691207
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["create", "local"]
	nodeLeafPlugs = ["create", "local"]
	pass

