

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	AbstractBaseCreate = Catalogue.AbstractBaseCreate
else:
	from .. import retriever
	AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
	assert AbstractBaseCreate

# add node doc



# region plug type defs

# endregion


# define node class
class AbstractBaseNurbsConversion(AbstractBaseCreate):

	# node attributes

	typeName = "abstractBaseNurbsConversion"
	typeIdInt = 1312968259
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

