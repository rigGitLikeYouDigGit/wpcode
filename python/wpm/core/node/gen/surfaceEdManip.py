

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ManipContainer = Catalogue.ManipContainer
else:
	from .. import retriever
	ManipContainer = retriever.getNodeCls("ManipContainer")
	assert ManipContainer

# add node doc



# region plug type defs

# endregion


# define node class
class SurfaceEdManip(ManipContainer):

	# node attributes

	typeName = "surfaceEdManip"
	apiTypeInt = 777
	apiTypeStr = "kSurfaceEdManip"
	typeIdInt = 1431130949
	MFnCls = om.MFnManip3D
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

