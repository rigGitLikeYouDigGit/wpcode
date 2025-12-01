

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Mesh = retriever.getNodeCls("Mesh")
assert Mesh
if T.TYPE_CHECKING:
	from .. import Mesh

# add node doc



# region plug type defs

# endregion


# define node class
class GreasePlaneRenderShape(Mesh):

	# node attributes

	typeName = "greasePlaneRenderShape"
	apiTypeInt = 1087
	apiTypeStr = "kGreasePlaneRenderShape"
	typeIdInt = 1196446291
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

