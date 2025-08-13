

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Field = retriever.getNodeCls("Field")
assert Field
if T.TYPE_CHECKING:
	from .. import Field

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
	pass

