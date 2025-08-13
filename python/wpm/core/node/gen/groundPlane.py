

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
SketchPlane = retriever.getNodeCls("SketchPlane")
assert SketchPlane
if T.TYPE_CHECKING:
	from .. import SketchPlane

# add node doc



# region plug type defs

# endregion


# define node class
class GroundPlane(SketchPlane):

	# node attributes

	typeName = "groundPlane"
	apiTypeInt = 290
	apiTypeStr = "kGroundPlane"
	typeIdInt = 1196444750
	MFnCls = om.MFnDagNode
	pass

