

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
class InPositionXPlug(Plug):
	parent : InPositionPlug = PlugDescriptor("inPosition")
	node : ClosestPointOnSurface = None
	pass
class InPositionYPlug(Plug):
	parent : InPositionPlug = PlugDescriptor("inPosition")
	node : ClosestPointOnSurface = None
	pass
class InPositionZPlug(Plug):
	parent : InPositionPlug = PlugDescriptor("inPosition")
	node : ClosestPointOnSurface = None
	pass
class InPositionPlug(Plug):
	inPositionX_ : InPositionXPlug = PlugDescriptor("inPositionX")
	ipx_ : InPositionXPlug = PlugDescriptor("inPositionX")
	inPositionY_ : InPositionYPlug = PlugDescriptor("inPositionY")
	ipy_ : InPositionYPlug = PlugDescriptor("inPositionY")
	inPositionZ_ : InPositionZPlug = PlugDescriptor("inPositionZ")
	ipz_ : InPositionZPlug = PlugDescriptor("inPositionZ")
	node : ClosestPointOnSurface = None
	pass
class InputSurfacePlug(Plug):
	node : ClosestPointOnSurface = None
	pass
class ParameterUPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	node : ClosestPointOnSurface = None
	pass
class ParameterVPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	node : ClosestPointOnSurface = None
	pass
class PositionXPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : ClosestPointOnSurface = None
	pass
class PositionYPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : ClosestPointOnSurface = None
	pass
class PositionZPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : ClosestPointOnSurface = None
	pass
class PositionPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	px_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	py_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	pz_ : PositionZPlug = PlugDescriptor("positionZ")
	node : ClosestPointOnSurface = None
	pass
class ResultPlug(Plug):
	parameterU_ : ParameterUPlug = PlugDescriptor("parameterU")
	u_ : ParameterUPlug = PlugDescriptor("parameterU")
	parameterV_ : ParameterVPlug = PlugDescriptor("parameterV")
	v_ : ParameterVPlug = PlugDescriptor("parameterV")
	position_ : PositionPlug = PlugDescriptor("position")
	p_ : PositionPlug = PlugDescriptor("position")
	node : ClosestPointOnSurface = None
	pass
# endregion


# define node class
class ClosestPointOnSurface(AbstractBaseCreate):
	inPositionX_ : InPositionXPlug = PlugDescriptor("inPositionX")
	inPositionY_ : InPositionYPlug = PlugDescriptor("inPositionY")
	inPositionZ_ : InPositionZPlug = PlugDescriptor("inPositionZ")
	inPosition_ : InPositionPlug = PlugDescriptor("inPosition")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	parameterU_ : ParameterUPlug = PlugDescriptor("parameterU")
	parameterV_ : ParameterVPlug = PlugDescriptor("parameterV")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	position_ : PositionPlug = PlugDescriptor("position")
	result_ : ResultPlug = PlugDescriptor("result")

	# node attributes

	typeName = "closestPointOnSurface"
	apiTypeInt = 56
	apiTypeStr = "kClosestPointOnSurface"
	typeIdInt = 1313034323
	MFnCls = om.MFnDependencyNode
	pass

