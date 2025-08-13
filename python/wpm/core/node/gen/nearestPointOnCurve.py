

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
	node : NearestPointOnCurve = None
	pass
class InPositionYPlug(Plug):
	parent : InPositionPlug = PlugDescriptor("inPosition")
	node : NearestPointOnCurve = None
	pass
class InPositionZPlug(Plug):
	parent : InPositionPlug = PlugDescriptor("inPosition")
	node : NearestPointOnCurve = None
	pass
class InPositionPlug(Plug):
	inPositionX_ : InPositionXPlug = PlugDescriptor("inPositionX")
	ipx_ : InPositionXPlug = PlugDescriptor("inPositionX")
	inPositionY_ : InPositionYPlug = PlugDescriptor("inPositionY")
	ipy_ : InPositionYPlug = PlugDescriptor("inPositionY")
	inPositionZ_ : InPositionZPlug = PlugDescriptor("inPositionZ")
	ipz_ : InPositionZPlug = PlugDescriptor("inPositionZ")
	node : NearestPointOnCurve = None
	pass
class InputCurvePlug(Plug):
	node : NearestPointOnCurve = None
	pass
class ParameterPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	node : NearestPointOnCurve = None
	pass
class PositionXPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : NearestPointOnCurve = None
	pass
class PositionYPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : NearestPointOnCurve = None
	pass
class PositionZPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : NearestPointOnCurve = None
	pass
class PositionPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	px_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	py_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	pz_ : PositionZPlug = PlugDescriptor("positionZ")
	node : NearestPointOnCurve = None
	pass
class ResultPlug(Plug):
	parameter_ : ParameterPlug = PlugDescriptor("parameter")
	pr_ : ParameterPlug = PlugDescriptor("parameter")
	position_ : PositionPlug = PlugDescriptor("position")
	p_ : PositionPlug = PlugDescriptor("position")
	node : NearestPointOnCurve = None
	pass
# endregion


# define node class
class NearestPointOnCurve(AbstractBaseCreate):
	inPositionX_ : InPositionXPlug = PlugDescriptor("inPositionX")
	inPositionY_ : InPositionYPlug = PlugDescriptor("inPositionY")
	inPositionZ_ : InPositionZPlug = PlugDescriptor("inPositionZ")
	inPosition_ : InPositionPlug = PlugDescriptor("inPosition")
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	parameter_ : ParameterPlug = PlugDescriptor("parameter")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	position_ : PositionPlug = PlugDescriptor("position")
	result_ : ResultPlug = PlugDescriptor("result")

	# node attributes

	typeName = "nearestPointOnCurve"
	apiTypeInt = 1065
	apiTypeStr = "kNearestPointOnCurve"
	typeIdInt = 1313886019
	MFnCls = om.MFnDependencyNode
	pass

