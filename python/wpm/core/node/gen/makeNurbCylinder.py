

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	RevolvedPrimitive = Catalogue.RevolvedPrimitive
else:
	from .. import retriever
	RevolvedPrimitive = retriever.getNodeCls("RevolvedPrimitive")
	assert RevolvedPrimitive

# add node doc



# region plug type defs

# endregion


# define node class
class MakeNurbCylinder(RevolvedPrimitive):

	# node attributes

	typeName = "makeNurbCylinder"
	typeIdInt = 1313036620
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

