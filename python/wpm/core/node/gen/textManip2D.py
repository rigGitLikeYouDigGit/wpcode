

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
class TextManip2D(Manip2D):

	# node attributes

	typeName = "textManip2D"
	typeIdInt = 1431131205
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

