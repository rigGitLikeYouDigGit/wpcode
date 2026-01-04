

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	BoundaryBase = Catalogue.BoundaryBase
else:
	from .. import retriever
	BoundaryBase = retriever.getNodeCls("BoundaryBase")
	assert BoundaryBase

# add node doc



# region plug type defs
class EndPointPlug(Plug):
	node : Boundary = None
	pass
class OrderPlug(Plug):
	node : Boundary = None
	pass
# endregion


# define node class
class Boundary(BoundaryBase):
	endPoint_ : EndPointPlug = PlugDescriptor("endPoint")
	order_ : OrderPlug = PlugDescriptor("order")

	# node attributes

	typeName = "boundary"
	apiTypeInt = 53
	apiTypeStr = "kBoundary"
	typeIdInt = 1312968260
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["endPoint", "order"]
	nodeLeafPlugs = ["endPoint", "order"]
	pass

