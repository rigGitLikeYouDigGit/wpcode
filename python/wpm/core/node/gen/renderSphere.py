

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ImplicitSphere = Catalogue.ImplicitSphere
else:
	from .. import retriever
	ImplicitSphere = retriever.getNodeCls("ImplicitSphere")
	assert ImplicitSphere

# add node doc



# region plug type defs

# endregion


# define node class
class RenderSphere(ImplicitSphere):

	# node attributes

	typeName = "renderSphere"
	apiTypeInt = 298
	apiTypeStr = "kRenderSphere"
	typeIdInt = 1380864848
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

