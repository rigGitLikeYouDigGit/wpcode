

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
	node : GeometryVarGroup = None
	pass
class LocalPlug(Plug):
	node : GeometryVarGroup = None
	pass
# endregion


# define node class
class GeometryVarGroup(BaseGeometryVarGroup):
	create_ : CreatePlug = PlugDescriptor("create")
	local_ : LocalPlug = PlugDescriptor("local")

	# node attributes

	typeName = "geometryVarGroup"
	typeIdInt = 1313297991
	pass

