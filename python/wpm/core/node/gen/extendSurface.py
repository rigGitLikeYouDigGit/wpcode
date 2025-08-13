

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
class DistancePlug(Plug):
	node : ExtendSurface = None
	pass
class ExtendDirectionPlug(Plug):
	node : ExtendSurface = None
	pass
class ExtendMethodPlug(Plug):
	node : ExtendSurface = None
	pass
class ExtendSidePlug(Plug):
	node : ExtendSurface = None
	pass
class ExtensionTypePlug(Plug):
	node : ExtendSurface = None
	pass
class InputSurfacePlug(Plug):
	node : ExtendSurface = None
	pass
class JoinPlug(Plug):
	node : ExtendSurface = None
	pass
class OutputSurfacePlug(Plug):
	node : ExtendSurface = None
	pass
class TargetObjectPlug(Plug):
	node : ExtendSurface = None
	pass
# endregion


# define node class
class ExtendSurface(AbstractBaseCreate):
	distance_ : DistancePlug = PlugDescriptor("distance")
	extendDirection_ : ExtendDirectionPlug = PlugDescriptor("extendDirection")
	extendMethod_ : ExtendMethodPlug = PlugDescriptor("extendMethod")
	extendSide_ : ExtendSidePlug = PlugDescriptor("extendSide")
	extensionType_ : ExtensionTypePlug = PlugDescriptor("extensionType")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	join_ : JoinPlug = PlugDescriptor("join")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	targetObject_ : TargetObjectPlug = PlugDescriptor("targetObject")

	# node attributes

	typeName = "extendSurface"
	apiTypeInt = 66
	apiTypeStr = "kExtendSurface"
	typeIdInt = 1313167443
	MFnCls = om.MFnDependencyNode
	pass

