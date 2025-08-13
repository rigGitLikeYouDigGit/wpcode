

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Camera = retriever.getNodeCls("Camera")
assert Camera
if T.TYPE_CHECKING:
	from .. import Camera

# add node doc



# region plug type defs

# endregion


# define node class
class UfeProxyCameraShape(Camera):

	# node attributes

	typeName = "ufeProxyCameraShape"
	typeIdInt = 1430671427
	pass

