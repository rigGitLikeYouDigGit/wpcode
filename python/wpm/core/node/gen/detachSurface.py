

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
class DirectionPlug(Plug):
	node : DetachSurface = None
	pass
class InputSurfacePlug(Plug):
	node : DetachSurface = None
	pass
class KeepPlug(Plug):
	node : DetachSurface = None
	pass
class OutputSurfacePlug(Plug):
	node : DetachSurface = None
	pass
class ParameterPlug(Plug):
	node : DetachSurface = None
	pass
# endregion


# define node class
class DetachSurface(AbstractBaseCreate):
	direction_ : DirectionPlug = PlugDescriptor("direction")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	keep_ : KeepPlug = PlugDescriptor("keep")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	parameter_ : ParameterPlug = PlugDescriptor("parameter")

	# node attributes

	typeName = "detachSurface"
	apiTypeInt = 64
	apiTypeStr = "kDetachSurface"
	typeIdInt = 1313100883
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["direction", "inputSurface", "keep", "outputSurface", "parameter"]
	nodeLeafPlugs = ["direction", "inputSurface", "keep", "outputSurface", "parameter"]
	pass

