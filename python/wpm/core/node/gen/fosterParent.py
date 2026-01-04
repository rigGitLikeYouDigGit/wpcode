

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

# endregion


# define node class
class FosterParent(Transform):

	# node attributes

	typeName = "fosterParent"
	apiTypeInt = 1092
	apiTypeStr = "kFosterParent"
	typeIdInt = 1179669070
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

