

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
	node : AttachSurface = None
	pass
class BlendKnotInsertionPlug(Plug):
	node : AttachSurface = None
	pass
class DirectionUPlug(Plug):
	node : AttachSurface = None
	pass
class InputSurface1Plug(Plug):
	node : AttachSurface = None
	pass
class InputSurface2Plug(Plug):
	node : AttachSurface = None
	pass
class KeepMultipleKnotsPlug(Plug):
	node : AttachSurface = None
	pass
class MethodPlug(Plug):
	node : AttachSurface = None
	pass
class OutputSurfacePlug(Plug):
	node : AttachSurface = None
	pass
class ParameterPlug(Plug):
	node : AttachSurface = None
	pass
class Reverse1Plug(Plug):
	node : AttachSurface = None
	pass
class Reverse2Plug(Plug):
	node : AttachSurface = None
	pass
class Swap1Plug(Plug):
	node : AttachSurface = None
	pass
class Swap2Plug(Plug):
	node : AttachSurface = None
	pass
class TwistPlug(Plug):
	node : AttachSurface = None
	pass
# endregion


# define node class
class AttachSurface(AbstractBaseCreate):
	blendBias_ : BlendBiasPlug = PlugDescriptor("blendBias")
	blendKnotInsertion_ : BlendKnotInsertionPlug = PlugDescriptor("blendKnotInsertion")
	directionU_ : DirectionUPlug = PlugDescriptor("directionU")
	inputSurface1_ : InputSurface1Plug = PlugDescriptor("inputSurface1")
	inputSurface2_ : InputSurface2Plug = PlugDescriptor("inputSurface2")
	keepMultipleKnots_ : KeepMultipleKnotsPlug = PlugDescriptor("keepMultipleKnots")
	method_ : MethodPlug = PlugDescriptor("method")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	parameter_ : ParameterPlug = PlugDescriptor("parameter")
	reverse1_ : Reverse1Plug = PlugDescriptor("reverse1")
	reverse2_ : Reverse2Plug = PlugDescriptor("reverse2")
	swap1_ : Swap1Plug = PlugDescriptor("swap1")
	swap2_ : Swap2Plug = PlugDescriptor("swap2")
	twist_ : TwistPlug = PlugDescriptor("twist")

	# node attributes

	typeName = "attachSurface"
	apiTypeInt = 44
	apiTypeStr = "kAttachSurface"
	typeIdInt = 1312904275
	MFnCls = om.MFnDependencyNode
	pass

