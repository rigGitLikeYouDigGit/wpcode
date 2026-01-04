

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	FlexorShape = Catalogue.FlexorShape
else:
	from .. import retriever
	FlexorShape = retriever.getNodeCls("FlexorShape")
	assert FlexorShape

# add node doc



# region plug type defs

# endregion


# define node class
class ClusterFlexorShape(FlexorShape):

	# node attributes

	typeName = "clusterFlexorShape"
	typeIdInt = 1179272006
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

