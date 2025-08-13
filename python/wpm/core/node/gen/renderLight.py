

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Light = retriever.getNodeCls("Light")
assert Light
if T.TYPE_CHECKING:
	from .. import Light

# add node doc



# region plug type defs
class RayInstancePlug(Plug):
	node : RenderLight = None
	pass
# endregion


# define node class
class RenderLight(Light):
	rayInstance_ : RayInstancePlug = PlugDescriptor("rayInstance")

	# node attributes

	typeName = "renderLight"
	typeIdInt = 1380733774
	pass

