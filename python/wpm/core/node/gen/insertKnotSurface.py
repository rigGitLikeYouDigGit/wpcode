

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
class AddKnotsPlug(Plug):
	node : InsertKnotSurface = None
	pass
class DirectionPlug(Plug):
	node : InsertKnotSurface = None
	pass
class InputSurfacePlug(Plug):
	node : InsertKnotSurface = None
	pass
class InsertBetweenPlug(Plug):
	node : InsertKnotSurface = None
	pass
class NumberOfKnotsPlug(Plug):
	node : InsertKnotSurface = None
	pass
class OutputSurfacePlug(Plug):
	node : InsertKnotSurface = None
	pass
class ParameterPlug(Plug):
	node : InsertKnotSurface = None
	pass
# endregion


# define node class
class InsertKnotSurface(AbstractBaseCreate):
	addKnots_ : AddKnotsPlug = PlugDescriptor("addKnots")
	direction_ : DirectionPlug = PlugDescriptor("direction")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	insertBetween_ : InsertBetweenPlug = PlugDescriptor("insertBetween")
	numberOfKnots_ : NumberOfKnotsPlug = PlugDescriptor("numberOfKnots")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	parameter_ : ParameterPlug = PlugDescriptor("parameter")

	# node attributes

	typeName = "insertKnotSurface"
	typeIdInt = 1313426259
	nodeLeafClassAttrs = ["addKnots", "direction", "inputSurface", "insertBetween", "numberOfKnots", "outputSurface", "parameter"]
	nodeLeafPlugs = ["addKnots", "direction", "inputSurface", "insertBetween", "numberOfKnots", "outputSurface", "parameter"]
	pass

