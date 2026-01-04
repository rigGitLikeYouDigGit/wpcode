

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Field = Catalogue.Field
else:
	from .. import retriever
	Field = retriever.getNodeCls("Field")
	assert Field

# add node doc



# region plug type defs
class RadialTypePlug(Plug):
	node : RadialField = None
	pass
# endregion


# define node class
class RadialField(Field):
	radialType_ : RadialTypePlug = PlugDescriptor("radialType")

	# node attributes

	typeName = "radialField"
	typeIdInt = 1498562884
	nodeLeafClassAttrs = ["radialType"]
	nodeLeafPlugs = ["radialType"]
	pass

