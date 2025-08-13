

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
class DistancePlug(Plug):
	node : OffsetSurface = None
	pass
class InputSurfacePlug(Plug):
	node : OffsetSurface = None
	pass
class MethodPlug(Plug):
	node : OffsetSurface = None
	pass
class OutputSurfacePlug(Plug):
	node : OffsetSurface = None
	pass
# endregion


# define node class
class OffsetSurface(AbstractBaseCreate):
	distance_ : DistancePlug = PlugDescriptor("distance")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	method_ : MethodPlug = PlugDescriptor("method")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")

	# node attributes

	typeName = "offsetSurface"
	apiTypeInt = 644
	apiTypeStr = "kOffsetSurface"
	typeIdInt = 1313821525
	MFnCls = om.MFnDependencyNode
	pass

