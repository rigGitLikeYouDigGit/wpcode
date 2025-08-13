

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
CurveFromSurface = retriever.getNodeCls("CurveFromSurface")
assert CurveFromSurface
if T.TYPE_CHECKING:
	from .. import CurveFromSurface

# add node doc



# region plug type defs
class BoundaryPlug(Plug):
	node : CurveFromSurfaceBnd = None
	pass
class EdgePlug(Plug):
	node : CurveFromSurfaceBnd = None
	pass
class FacePlug(Plug):
	node : CurveFromSurfaceBnd = None
	pass
# endregion


# define node class
class CurveFromSurfaceBnd(CurveFromSurface):
	boundary_ : BoundaryPlug = PlugDescriptor("boundary")
	edge_ : EdgePlug = PlugDescriptor("edge")
	face_ : FacePlug = PlugDescriptor("face")

	# node attributes

	typeName = "curveFromSurfaceBnd"
	apiTypeInt = 59
	apiTypeStr = "kCurveFromSurfaceBnd"
	typeIdInt = 1313035074
	MFnCls = om.MFnDependencyNode
	pass

