

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
class MinDistancePlug(Plug):
	node : NewtonField = None
	pass
class OwnerMassDataPlug(Plug):
	node : NewtonField = None
	pass
# endregion


# define node class
class NewtonField(Field):
	minDistance_ : MinDistancePlug = PlugDescriptor("minDistance")
	ownerMassData_ : OwnerMassDataPlug = PlugDescriptor("ownerMassData")

	# node attributes

	typeName = "newtonField"
	typeIdInt = 1498301783
	pass

