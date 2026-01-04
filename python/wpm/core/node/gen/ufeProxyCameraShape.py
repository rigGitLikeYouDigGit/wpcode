

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Camera = Catalogue.Camera
else:
	from .. import retriever
	Camera = retriever.getNodeCls("Camera")
	assert Camera

# add node doc



# region plug type defs

# endregion


# define node class
class UfeProxyCameraShape(Camera):

	# node attributes

	typeName = "ufeProxyCameraShape"
	typeIdInt = 1430671427
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

