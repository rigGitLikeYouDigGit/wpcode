

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	AbstractBaseCreate = Catalogue.AbstractBaseCreate
else:
	from .. import retriever
	AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
	assert AbstractBaseCreate

# add node doc



# region plug type defs
class CurveOnSurface1Plug(Plug):
	node : IntersectSurface = None
	pass
class CurveOnSurface2Plug(Plug):
	node : IntersectSurface = None
	pass
class InputSurface1Plug(Plug):
	node : IntersectSurface = None
	pass
class InputSurface2Plug(Plug):
	node : IntersectSurface = None
	pass
class Output3dCurvePlug(Plug):
	node : IntersectSurface = None
	pass
class TolerancePlug(Plug):
	node : IntersectSurface = None
	pass
# endregion


# define node class
class IntersectSurface(AbstractBaseCreate):
	curveOnSurface1_ : CurveOnSurface1Plug = PlugDescriptor("curveOnSurface1")
	curveOnSurface2_ : CurveOnSurface2Plug = PlugDescriptor("curveOnSurface2")
	inputSurface1_ : InputSurface1Plug = PlugDescriptor("inputSurface1")
	inputSurface2_ : InputSurface2Plug = PlugDescriptor("inputSurface2")
	output3dCurve_ : Output3dCurvePlug = PlugDescriptor("output3dCurve")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")

	# node attributes

	typeName = "intersectSurface"
	apiTypeInt = 77
	apiTypeStr = "kIntersectSurface"
	typeIdInt = 1313428294
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["curveOnSurface1", "curveOnSurface2", "inputSurface1", "inputSurface2", "output3dCurve", "tolerance"]
	nodeLeafPlugs = ["curveOnSurface1", "curveOnSurface2", "inputSurface1", "inputSurface2", "output3dCurve", "tolerance"]
	pass

