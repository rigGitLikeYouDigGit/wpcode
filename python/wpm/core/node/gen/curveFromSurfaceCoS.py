

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	CurveFromSurface = Catalogue.CurveFromSurface
else:
	from .. import retriever
	CurveFromSurface = retriever.getNodeCls("CurveFromSurface")
	assert CurveFromSurface

# add node doc



# region plug type defs
class CurveOnSurfacePlug(Plug):
	node : CurveFromSurfaceCoS = None
	pass
# endregion


# define node class
class CurveFromSurfaceCoS(CurveFromSurface):
	curveOnSurface_ : CurveOnSurfacePlug = PlugDescriptor("curveOnSurface")

	# node attributes

	typeName = "curveFromSurfaceCoS"
	apiTypeInt = 60
	apiTypeStr = "kCurveFromSurfaceCoS"
	typeIdInt = 1313035075
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["curveOnSurface"]
	nodeLeafPlugs = ["curveOnSurface"]
	pass

