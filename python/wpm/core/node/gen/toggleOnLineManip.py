

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	GeometryOnLineManip = Catalogue.GeometryOnLineManip
else:
	from .. import retriever
	GeometryOnLineManip = retriever.getNodeCls("GeometryOnLineManip")
	assert GeometryOnLineManip

# add node doc



# region plug type defs

# endregion


# define node class
class ToggleOnLineManip(GeometryOnLineManip):

	# node attributes

	typeName = "toggleOnLineManip"
	apiTypeInt = 144
	apiTypeStr = "kToggleOnLineManip"
	typeIdInt = 1431588684
	MFnCls = om.MFnManip3D
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

