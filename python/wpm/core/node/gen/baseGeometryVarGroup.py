

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Transform = Catalogue.Transform
else:
	from .. import retriever
	Transform = retriever.getNodeCls("Transform")
	assert Transform

# add node doc



# region plug type defs
class MaxCreatedPlug(Plug):
	node : BaseGeometryVarGroup = None
	pass
# endregion


# define node class
class BaseGeometryVarGroup(Transform):
	maxCreated_ : MaxCreatedPlug = PlugDescriptor("maxCreated")

	# node attributes

	typeName = "baseGeometryVarGroup"
	typeIdInt = 1312970311
	nodeLeafClassAttrs = ["maxCreated"]
	nodeLeafPlugs = ["maxCreated"]
	pass

