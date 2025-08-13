

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
	node : CloseCurve = None
	pass
class BlendKnotInsertionPlug(Plug):
	node : CloseCurve = None
	pass
class InputCurvePlug(Plug):
	node : CloseCurve = None
	pass
class OutputCurvePlug(Plug):
	node : CloseCurve = None
	pass
class ParameterPlug(Plug):
	node : CloseCurve = None
	pass
class PreserveShapePlug(Plug):
	node : CloseCurve = None
	pass
# endregion


# define node class
class CloseCurve(AbstractBaseCreate):
	blendBias_ : BlendBiasPlug = PlugDescriptor("blendBias")
	blendKnotInsertion_ : BlendKnotInsertionPlug = PlugDescriptor("blendKnotInsertion")
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	parameter_ : ParameterPlug = PlugDescriptor("parameter")
	preserveShape_ : PreserveShapePlug = PlugDescriptor("preserveShape")

	# node attributes

	typeName = "closeCurve"
	apiTypeInt = 55
	apiTypeStr = "kCloseCurve"
	typeIdInt = 1313030997
	MFnCls = om.MFnDependencyNode
	pass

