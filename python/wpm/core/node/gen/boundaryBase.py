

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AbstractBaseCreate = retriever.getNodeCls("AbstractBaseCreate")
assert AbstractBaseCreate
if T.TYPE_CHECKING:
	from .. import AbstractBaseCreate

# add node doc



# region plug type defs
class EndPointTolerancePlug(Plug):
	node : BoundaryBase = None
	pass
class InputCurve1Plug(Plug):
	node : BoundaryBase = None
	pass
class InputCurve2Plug(Plug):
	node : BoundaryBase = None
	pass
class InputCurve3Plug(Plug):
	node : BoundaryBase = None
	pass
class InputCurve4Plug(Plug):
	node : BoundaryBase = None
	pass
class OutputSurfacePlug(Plug):
	node : BoundaryBase = None
	pass
# endregion


# define node class
class BoundaryBase(AbstractBaseCreate):
	endPointTolerance_ : EndPointTolerancePlug = PlugDescriptor("endPointTolerance")
	inputCurve1_ : InputCurve1Plug = PlugDescriptor("inputCurve1")
	inputCurve2_ : InputCurve2Plug = PlugDescriptor("inputCurve2")
	inputCurve3_ : InputCurve3Plug = PlugDescriptor("inputCurve3")
	inputCurve4_ : InputCurve4Plug = PlugDescriptor("inputCurve4")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")

	# node attributes

	typeName = "boundaryBase"
	typeIdInt = 1312965203
	nodeLeafClassAttrs = ["endPointTolerance", "inputCurve1", "inputCurve2", "inputCurve3", "inputCurve4", "outputSurface"]
	nodeLeafPlugs = ["endPointTolerance", "inputCurve1", "inputCurve2", "inputCurve3", "inputCurve4", "outputSurface"]
	pass

