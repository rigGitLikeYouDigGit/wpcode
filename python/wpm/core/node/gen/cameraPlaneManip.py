

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
GeometryOnLineManip = retriever.getNodeCls("GeometryOnLineManip")
assert GeometryOnLineManip
if T.TYPE_CHECKING:
	from .. import GeometryOnLineManip

# add node doc



# region plug type defs

# endregion


# define node class
class CameraPlaneManip(GeometryOnLineManip):

	# node attributes

	typeName = "cameraPlaneManip"
	apiTypeInt = 143
	apiTypeStr = "kCameraPlaneManip"
	typeIdInt = 1431126864
	MFnCls = om.MFnManip3D
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

