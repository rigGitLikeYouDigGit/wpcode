

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
class AttachPlug(Plug):
	node : AlignCurve = None
	pass
class CurvatureContinuityPlug(Plug):
	node : AlignCurve = None
	pass
class CurvatureScale1Plug(Plug):
	node : AlignCurve = None
	pass
class CurvatureScale2Plug(Plug):
	node : AlignCurve = None
	pass
class InputCurve1Plug(Plug):
	node : AlignCurve = None
	pass
class InputCurve2Plug(Plug):
	node : AlignCurve = None
	pass
class JoinParameterPlug(Plug):
	node : AlignCurve = None
	pass
class KeepMultipleKnotsPlug(Plug):
	node : AlignCurve = None
	pass
class OutputCurve1Plug(Plug):
	node : AlignCurve = None
	pass
class OutputCurve2Plug(Plug):
	node : AlignCurve = None
	pass
class PositionalContinuityPlug(Plug):
	node : AlignCurve = None
	pass
class PositionalContinuityTypePlug(Plug):
	node : AlignCurve = None
	pass
class Reverse1Plug(Plug):
	node : AlignCurve = None
	pass
class Reverse2Plug(Plug):
	node : AlignCurve = None
	pass
class TangentContinuityPlug(Plug):
	node : AlignCurve = None
	pass
class TangentContinuityTypePlug(Plug):
	node : AlignCurve = None
	pass
class TangentScale1Plug(Plug):
	node : AlignCurve = None
	pass
class TangentScale2Plug(Plug):
	node : AlignCurve = None
	pass
# endregion


# define node class
class AlignCurve(AbstractBaseCreate):
	attach_ : AttachPlug = PlugDescriptor("attach")
	curvatureContinuity_ : CurvatureContinuityPlug = PlugDescriptor("curvatureContinuity")
	curvatureScale1_ : CurvatureScale1Plug = PlugDescriptor("curvatureScale1")
	curvatureScale2_ : CurvatureScale2Plug = PlugDescriptor("curvatureScale2")
	inputCurve1_ : InputCurve1Plug = PlugDescriptor("inputCurve1")
	inputCurve2_ : InputCurve2Plug = PlugDescriptor("inputCurve2")
	joinParameter_ : JoinParameterPlug = PlugDescriptor("joinParameter")
	keepMultipleKnots_ : KeepMultipleKnotsPlug = PlugDescriptor("keepMultipleKnots")
	outputCurve1_ : OutputCurve1Plug = PlugDescriptor("outputCurve1")
	outputCurve2_ : OutputCurve2Plug = PlugDescriptor("outputCurve2")
	positionalContinuity_ : PositionalContinuityPlug = PlugDescriptor("positionalContinuity")
	positionalContinuityType_ : PositionalContinuityTypePlug = PlugDescriptor("positionalContinuityType")
	reverse1_ : Reverse1Plug = PlugDescriptor("reverse1")
	reverse2_ : Reverse2Plug = PlugDescriptor("reverse2")
	tangentContinuity_ : TangentContinuityPlug = PlugDescriptor("tangentContinuity")
	tangentContinuityType_ : TangentContinuityTypePlug = PlugDescriptor("tangentContinuityType")
	tangentScale1_ : TangentScale1Plug = PlugDescriptor("tangentScale1")
	tangentScale2_ : TangentScale2Plug = PlugDescriptor("tangentScale2")

	# node attributes

	typeName = "alignCurve"
	apiTypeInt = 41
	apiTypeStr = "kAlignCurve"
	typeIdInt = 1312902211
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["attach", "curvatureContinuity", "curvatureScale1", "curvatureScale2", "inputCurve1", "inputCurve2", "joinParameter", "keepMultipleKnots", "outputCurve1", "outputCurve2", "positionalContinuity", "positionalContinuityType", "reverse1", "reverse2", "tangentContinuity", "tangentContinuityType", "tangentScale1", "tangentScale2"]
	nodeLeafPlugs = ["attach", "curvatureContinuity", "curvatureScale1", "curvatureScale2", "inputCurve1", "inputCurve2", "joinParameter", "keepMultipleKnots", "outputCurve1", "outputCurve2", "positionalContinuity", "positionalContinuityType", "reverse1", "reverse2", "tangentContinuity", "tangentContinuityType", "tangentScale1", "tangentScale2"]
	pass

