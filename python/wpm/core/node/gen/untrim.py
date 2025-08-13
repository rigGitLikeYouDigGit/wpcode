

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
class InputSurfacePlug(Plug):
	node : Untrim = None
	pass
class OutputCurvePlug(Plug):
	node : Untrim = None
	pass
class OutputSurfacePlug(Plug):
	node : Untrim = None
	pass
# endregion


# define node class
class Untrim(AbstractBaseCreate):
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")

	# node attributes

	typeName = "untrim"
	apiTypeInt = 106
	apiTypeStr = "kUntrim"
	typeIdInt = 1314214994
	MFnCls = om.MFnDependencyNode
	pass

