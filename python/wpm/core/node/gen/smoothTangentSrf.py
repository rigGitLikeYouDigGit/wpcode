

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
class DirectionPlug(Plug):
	node : SmoothTangentSrf = None
	pass
class InputSurfacePlug(Plug):
	node : SmoothTangentSrf = None
	pass
class OutputSurfacePlug(Plug):
	node : SmoothTangentSrf = None
	pass
class ParameterPlug(Plug):
	node : SmoothTangentSrf = None
	pass
class SmoothnessPlug(Plug):
	node : SmoothTangentSrf = None
	pass
# endregion


# define node class
class SmoothTangentSrf(AbstractBaseCreate):
	direction_ : DirectionPlug = PlugDescriptor("direction")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	parameter_ : ParameterPlug = PlugDescriptor("parameter")
	smoothness_ : SmoothnessPlug = PlugDescriptor("smoothness")

	# node attributes

	typeName = "smoothTangentSrf"
	apiTypeInt = 782
	apiTypeStr = "kSmoothTangentSrf"
	typeIdInt = 1314083918
	MFnCls = om.MFnDependencyNode
	pass

