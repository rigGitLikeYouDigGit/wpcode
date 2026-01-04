

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	SubdModifier = Catalogue.SubdModifier
else:
	from .. import retriever
	SubdModifier = retriever.getNodeCls("SubdModifier")
	assert SubdModifier

# add node doc



# region plug type defs

# endregion


# define node class
class SubdMapCut(SubdModifier):

	# node attributes

	typeName = "subdMapCut"
	apiTypeInt = 872
	apiTypeStr = "kSubdMapCut"
	typeIdInt = 1397571907
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

