

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Plane = retriever.getNodeCls("Plane")
assert Plane
if T.TYPE_CHECKING:
	from .. import Plane

# add node doc



# region plug type defs

# endregion


# define node class
class SketchPlane(Plane):

	# node attributes

	typeName = "sketchPlane"
	apiTypeInt = 289
	apiTypeStr = "kSketchPlane"
	typeIdInt = 1397444686
	MFnCls = om.MFnDagNode
	pass

