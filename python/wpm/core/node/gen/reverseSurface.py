

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
	node : ReverseSurface = None
	pass
class InputSurfacePlug(Plug):
	node : ReverseSurface = None
	pass
class OutputSurfacePlug(Plug):
	node : ReverseSurface = None
	pass
# endregion


# define node class
class ReverseSurface(AbstractBaseCreate):
	direction_ : DirectionPlug = PlugDescriptor("direction")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")

	# node attributes

	typeName = "reverseSurface"
	apiTypeInt = 93
	apiTypeStr = "kReverseSurface"
	typeIdInt = 1314018899
	MFnCls = om.MFnDependencyNode
	pass

