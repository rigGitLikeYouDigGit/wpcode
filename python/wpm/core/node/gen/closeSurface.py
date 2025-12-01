

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
class BlendBiasPlug(Plug):
	node : CloseSurface = None
	pass
class BlendKnotInsertionPlug(Plug):
	node : CloseSurface = None
	pass
class DirectionPlug(Plug):
	node : CloseSurface = None
	pass
class InputSurfacePlug(Plug):
	node : CloseSurface = None
	pass
class OutputSurfacePlug(Plug):
	node : CloseSurface = None
	pass
class ParameterPlug(Plug):
	node : CloseSurface = None
	pass
class PreserveShapePlug(Plug):
	node : CloseSurface = None
	pass
# endregion


# define node class
class CloseSurface(AbstractBaseCreate):
	blendBias_ : BlendBiasPlug = PlugDescriptor("blendBias")
	blendKnotInsertion_ : BlendKnotInsertionPlug = PlugDescriptor("blendKnotInsertion")
	direction_ : DirectionPlug = PlugDescriptor("direction")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	parameter_ : ParameterPlug = PlugDescriptor("parameter")
	preserveShape_ : PreserveShapePlug = PlugDescriptor("preserveShape")

	# node attributes

	typeName = "closeSurface"
	apiTypeInt = 57
	apiTypeStr = "kCloseSurface"
	typeIdInt = 1313035093
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["blendBias", "blendKnotInsertion", "direction", "inputSurface", "outputSurface", "parameter", "preserveShape"]
	nodeLeafPlugs = ["blendBias", "blendKnotInsertion", "direction", "inputSurface", "outputSurface", "parameter", "preserveShape"]
	pass

