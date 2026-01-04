

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ContainerBase = Catalogue.ContainerBase
else:
	from .. import retriever
	ContainerBase = retriever.getNodeCls("ContainerBase")
	assert ContainerBase

# add node doc



# region plug type defs

# endregion


# define node class
class Entity(ContainerBase):

	# node attributes

	typeName = "entity"
	typeIdInt = 1162761305
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

