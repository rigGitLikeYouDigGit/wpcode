

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Plane = Catalogue.Plane
else:
	from .. import retriever
	Plane = retriever.getNodeCls("Plane")
	assert Plane

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
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

