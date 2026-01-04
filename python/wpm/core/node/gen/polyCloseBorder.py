

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
class PolyCloseBorder(PolyModifier):

	# node attributes

	typeName = "polyCloseBorder"
	apiTypeInt = 405
	apiTypeStr = "kPolyCloseBorder"
	typeIdInt = 1346587727
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

