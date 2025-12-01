

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
class OrthoGrid(SketchPlane):

	# node attributes

	typeName = "orthoGrid"
	apiTypeInt = 291
	apiTypeStr = "kOrthoGrid"
	typeIdInt = 1330792260
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

