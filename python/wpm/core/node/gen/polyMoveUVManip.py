

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Manip2DContainer = Catalogue.Manip2DContainer
else:
	from .. import retriever
	Manip2DContainer = retriever.getNodeCls("Manip2DContainer")
	assert Manip2DContainer

# add node doc



# region plug type defs

# endregion


# define node class
class PolyMoveUVManip(Manip2DContainer):

	# node attributes

	typeName = "polyMoveUVManip"
	apiTypeInt = 193
	apiTypeStr = "kPolyMoveUVManip"
	typeIdInt = 1431328077
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

