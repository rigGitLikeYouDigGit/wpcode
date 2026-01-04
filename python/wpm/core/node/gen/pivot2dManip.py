

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Manip2D = Catalogue.Manip2D
else:
	from .. import retriever
	Manip2D = retriever.getNodeCls("Manip2D")
	assert Manip2D

# add node doc



# region plug type defs

# endregion


# define node class
class Pivot2dManip(Manip2D):

	# node attributes

	typeName = "pivot2dManip"
	typeIdInt = 1431319117
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

