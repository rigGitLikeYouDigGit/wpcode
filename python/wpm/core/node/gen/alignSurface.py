

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
	node : AlignSurface = None
	pass
class CurvatureContinuityPlug(Plug):
	node : AlignSurface = None
	pass
class CurvatureScale1Plug(Plug):
	node : AlignSurface = None
	pass
class CurvatureScale2Plug(Plug):
	node : AlignSurface = None
	pass
class DirectionUPlug(Plug):
	node : AlignSurface = None
	pass
class InputSurface1Plug(Plug):
	node : AlignSurface = None
	pass
class InputSurface2Plug(Plug):
	node : AlignSurface = None
	pass
class JoinParameterPlug(Plug):
	node : AlignSurface = None
	pass
class KeepMultipleKnotsPlug(Plug):
	node : AlignSurface = None
	pass
class OutputSurface1Plug(Plug):
	node : AlignSurface = None
	pass
class OutputSurface2Plug(Plug):
	node : AlignSurface = None
	pass
class PositionalContinuityPlug(Plug):
	node : AlignSurface = None
	pass
class PositionalContinuityTypePlug(Plug):
	node : AlignSurface = None
	pass
class Reverse1Plug(Plug):
	node : AlignSurface = None
	pass
class Reverse2Plug(Plug):
	node : AlignSurface = None
	pass
class Swap1Plug(Plug):
	node : AlignSurface = None
	pass
class Swap2Plug(Plug):
	node : AlignSurface = None
	pass
class TangentContinuityPlug(Plug):
	node : AlignSurface = None
	pass
class TangentContinuityTypePlug(Plug):
	node : AlignSurface = None
	pass
class TangentScale1Plug(Plug):
	node : AlignSurface = None
	pass
class TangentScale2Plug(Plug):
	node : AlignSurface = None
	pass
class TwistPlug(Plug):
	node : AlignSurface = None
	pass
# endregion


# define node class
class AlignSurface(AbstractBaseCreate):
	attach_ : AttachPlug = PlugDescriptor("attach")
	curvatureContinuity_ : CurvatureContinuityPlug = PlugDescriptor("curvatureContinuity")
	curvatureScale1_ : CurvatureScale1Plug = PlugDescriptor("curvatureScale1")
	curvatureScale2_ : CurvatureScale2Plug = PlugDescriptor("curvatureScale2")
	directionU_ : DirectionUPlug = PlugDescriptor("directionU")
	inputSurface1_ : InputSurface1Plug = PlugDescriptor("inputSurface1")
	inputSurface2_ : InputSurface2Plug = PlugDescriptor("inputSurface2")
	joinParameter_ : JoinParameterPlug = PlugDescriptor("joinParameter")
	keepMultipleKnots_ : KeepMultipleKnotsPlug = PlugDescriptor("keepMultipleKnots")
	outputSurface1_ : OutputSurface1Plug = PlugDescriptor("outputSurface1")
	outputSurface2_ : OutputSurface2Plug = PlugDescriptor("outputSurface2")
	positionalContinuity_ : PositionalContinuityPlug = PlugDescriptor("positionalContinuity")
	positionalContinuityType_ : PositionalContinuityTypePlug = PlugDescriptor("positionalContinuityType")
	reverse1_ : Reverse1Plug = PlugDescriptor("reverse1")
	reverse2_ : Reverse2Plug = PlugDescriptor("reverse2")
	swap1_ : Swap1Plug = PlugDescriptor("swap1")
	swap2_ : Swap2Plug = PlugDescriptor("swap2")
	tangentContinuity_ : TangentContinuityPlug = PlugDescriptor("tangentContinuity")
	tangentContinuityType_ : TangentContinuityTypePlug = PlugDescriptor("tangentContinuityType")
	tangentScale1_ : TangentScale1Plug = PlugDescriptor("tangentScale1")
	tangentScale2_ : TangentScale2Plug = PlugDescriptor("tangentScale2")
	twist_ : TwistPlug = PlugDescriptor("twist")

	# node attributes

	typeName = "alignSurface"
	apiTypeInt = 42
	apiTypeStr = "kAlignSurface"
	typeIdInt = 1312902227
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["attach", "curvatureContinuity", "curvatureScale1", "curvatureScale2", "directionU", "inputSurface1", "inputSurface2", "joinParameter", "keepMultipleKnots", "outputSurface1", "outputSurface2", "positionalContinuity", "positionalContinuityType", "reverse1", "reverse2", "swap1", "swap2", "tangentContinuity", "tangentContinuityType", "tangentScale1", "tangentScale2", "twist"]
	nodeLeafPlugs = ["attach", "curvatureContinuity", "curvatureScale1", "curvatureScale2", "directionU", "inputSurface1", "inputSurface2", "joinParameter", "keepMultipleKnots", "outputSurface1", "outputSurface2", "positionalContinuity", "positionalContinuityType", "reverse1", "reverse2", "swap1", "swap2", "tangentContinuity", "tangentContinuityType", "tangentScale1", "tangentScale2", "twist"]
	pass

