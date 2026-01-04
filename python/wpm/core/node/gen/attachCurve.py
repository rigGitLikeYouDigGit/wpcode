

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
class BlendBiasPlug(Plug):
	node : AttachCurve = None
	pass
class BlendKnotInsertionPlug(Plug):
	node : AttachCurve = None
	pass
class InputCurve1Plug(Plug):
	node : AttachCurve = None
	pass
class InputCurve2Plug(Plug):
	node : AttachCurve = None
	pass
class InputCurvesPlug(Plug):
	node : AttachCurve = None
	pass
class KeepMultipleKnotsPlug(Plug):
	node : AttachCurve = None
	pass
class MethodPlug(Plug):
	node : AttachCurve = None
	pass
class OutputCurvePlug(Plug):
	node : AttachCurve = None
	pass
class ParameterPlug(Plug):
	node : AttachCurve = None
	pass
class Reverse1Plug(Plug):
	node : AttachCurve = None
	pass
class Reverse2Plug(Plug):
	node : AttachCurve = None
	pass
# endregion


# define node class
class AttachCurve(AbstractBaseCreate):
	blendBias_ : BlendBiasPlug = PlugDescriptor("blendBias")
	blendKnotInsertion_ : BlendKnotInsertionPlug = PlugDescriptor("blendKnotInsertion")
	inputCurve1_ : InputCurve1Plug = PlugDescriptor("inputCurve1")
	inputCurve2_ : InputCurve2Plug = PlugDescriptor("inputCurve2")
	inputCurves_ : InputCurvesPlug = PlugDescriptor("inputCurves")
	keepMultipleKnots_ : KeepMultipleKnotsPlug = PlugDescriptor("keepMultipleKnots")
	method_ : MethodPlug = PlugDescriptor("method")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	parameter_ : ParameterPlug = PlugDescriptor("parameter")
	reverse1_ : Reverse1Plug = PlugDescriptor("reverse1")
	reverse2_ : Reverse2Plug = PlugDescriptor("reverse2")

	# node attributes

	typeName = "attachCurve"
	apiTypeInt = 43
	apiTypeStr = "kAttachCurve"
	typeIdInt = 1312904259
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["blendBias", "blendKnotInsertion", "inputCurve1", "inputCurve2", "inputCurves", "keepMultipleKnots", "method", "outputCurve", "parameter", "reverse1", "reverse2"]
	nodeLeafPlugs = ["blendBias", "blendKnotInsertion", "inputCurve1", "inputCurve2", "inputCurves", "keepMultipleKnots", "method", "outputCurve", "parameter", "reverse1", "reverse2"]
	pass

