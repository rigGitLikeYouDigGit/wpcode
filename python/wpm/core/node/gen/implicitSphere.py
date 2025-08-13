

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
class RadiusPlug(Plug):
	node : ImplicitSphere = None
	pass
class SpherePlug(Plug):
	node : ImplicitSphere = None
	pass
# endregion


# define node class
class ImplicitSphere(GeometryShape):
	radius_ : RadiusPlug = PlugDescriptor("radius")
	sphere_ : SpherePlug = PlugDescriptor("sphere")

	# node attributes

	typeName = "implicitSphere"
	apiTypeInt = 895
	apiTypeStr = "kImplicitSphere"
	typeIdInt = 1179210576
	MFnCls = om.MFnDagNode
	pass

