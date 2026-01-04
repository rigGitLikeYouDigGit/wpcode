

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyMoveUV = Catalogue.PolyMoveUV
else:
	from .. import retriever
	PolyMoveUV = retriever.getNodeCls("PolyMoveUV")
	assert PolyMoveUV

# add node doc



# region plug type defs

# endregion


# define node class
class PolyMoveFacetUV(PolyMoveUV):

	# node attributes

	typeName = "polyMoveFacetUV"
	apiTypeInt = 420
	apiTypeStr = "kPolyMoveFacetUV"
	typeIdInt = 1347241557
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

