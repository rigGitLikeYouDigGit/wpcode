

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
GeometryShape = retriever.getNodeCls("GeometryShape")
assert GeometryShape
if T.TYPE_CHECKING:
	from .. import GeometryShape

# add node doc



# region plug type defs
class ConePlug(Plug):
	node : ImplicitCone = None
	pass
class ConeAnglePlug(Plug):
	node : ImplicitCone = None
	pass
class ConeCapPlug(Plug):
	node : ImplicitCone = None
	pass
# endregion


# define node class
class ImplicitCone(GeometryShape):
	cone_ : ConePlug = PlugDescriptor("cone")
	coneAngle_ : ConeAnglePlug = PlugDescriptor("coneAngle")
	coneCap_ : ConeCapPlug = PlugDescriptor("coneCap")

	# node attributes

	typeName = "implicitCone"
	apiTypeInt = 894
	apiTypeStr = "kImplicitCone"
	typeIdInt = 1179206479
	MFnCls = om.MFnDagNode
	pass

