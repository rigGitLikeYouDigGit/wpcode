

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	FieldManip = Catalogue.FieldManip
else:
	from .. import retriever
	FieldManip = retriever.getNodeCls("FieldManip")
	assert FieldManip

# add node doc



# region plug type defs

# endregion


# define node class
class NewtonManip(FieldManip):

	# node attributes

	typeName = "newtonManip"
	typeIdInt = 1430539853
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

