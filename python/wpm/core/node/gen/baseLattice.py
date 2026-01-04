

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Shape = Catalogue.Shape
else:
	from .. import retriever
	Shape = retriever.getNodeCls("Shape")
	assert Shape

# add node doc



# region plug type defs

# endregion


# define node class
class BaseLattice(Shape):

	# node attributes

	typeName = "baseLattice"
	apiTypeInt = 249
	apiTypeStr = "kBaseLattice"
	typeIdInt = 1178747219
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

