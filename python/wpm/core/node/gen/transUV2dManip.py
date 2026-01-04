

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Uv2dManip = Catalogue.Uv2dManip
else:
	from .. import retriever
	Uv2dManip = retriever.getNodeCls("Uv2dManip")
	assert Uv2dManip

# add node doc



# region plug type defs

# endregion


# define node class
class TransUV2dManip(Uv2dManip):

	# node attributes

	typeName = "transUV2dManip"
	typeIdInt = 1429361746
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

