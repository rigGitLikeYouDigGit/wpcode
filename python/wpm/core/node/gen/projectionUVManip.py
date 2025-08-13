

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ProjectionManip = retriever.getNodeCls("ProjectionManip")
assert ProjectionManip
if T.TYPE_CHECKING:
	from .. import ProjectionManip

# add node doc



# region plug type defs

# endregion


# define node class
class ProjectionUVManip(ProjectionManip):

	# node attributes

	typeName = "projectionUVManip"
	apiTypeInt = 175
	apiTypeStr = "kProjectionUVManip"
	typeIdInt = 1431130197
	MFnCls = om.MFnManip3D
	pass

