

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
class PolyDelFacet(PolyModifier):

	# node attributes

	typeName = "polyDelFacet"
	apiTypeInt = 410
	apiTypeStr = "kPolyDelFacet"
	typeIdInt = 1346651462
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

