

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
	node : InsertKnotCurve = None
	pass
class InputCurvePlug(Plug):
	node : InsertKnotCurve = None
	pass
class InsertBetweenPlug(Plug):
	node : InsertKnotCurve = None
	pass
class NumberOfKnotsPlug(Plug):
	node : InsertKnotCurve = None
	pass
class OutputCurvePlug(Plug):
	node : InsertKnotCurve = None
	pass
class ParameterPlug(Plug):
	node : InsertKnotCurve = None
	pass
# endregion


# define node class
class InsertKnotCurve(AbstractBaseCreate):
	addKnots_ : AddKnotsPlug = PlugDescriptor("addKnots")
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	insertBetween_ : InsertBetweenPlug = PlugDescriptor("insertBetween")
	numberOfKnots_ : NumberOfKnotsPlug = PlugDescriptor("numberOfKnots")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	parameter_ : ParameterPlug = PlugDescriptor("parameter")

	# node attributes

	typeName = "insertKnotCurve"
	typeIdInt = 1313426243
	nodeLeafClassAttrs = ["addKnots", "inputCurve", "insertBetween", "numberOfKnots", "outputCurve", "parameter"]
	nodeLeafPlugs = ["addKnots", "inputCurve", "insertBetween", "numberOfKnots", "outputCurve", "parameter"]
	pass

