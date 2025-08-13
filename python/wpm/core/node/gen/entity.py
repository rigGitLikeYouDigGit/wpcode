

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ContainerBase = retriever.getNodeCls("ContainerBase")
assert ContainerBase
if T.TYPE_CHECKING:
	from .. import ContainerBase

# add node doc



# region plug type defs

# endregion


# define node class
class Entity(ContainerBase):

	# node attributes

	typeName = "entity"
	typeIdInt = 1162761305
	pass

