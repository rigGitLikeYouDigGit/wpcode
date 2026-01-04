

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifier = Catalogue.PolyModifier
else:
	from .. import retriever
	PolyModifier = retriever.getNodeCls("PolyModifier")
	assert PolyModifier

# add node doc



# region plug type defs

# endregion


# define node class
class PolyDelVertex(PolyModifier):

	# node attributes

	typeName = "polyDelVertex"
	apiTypeInt = 411
	apiTypeStr = "kPolyDelVertex"
	typeIdInt = 1346651478
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

