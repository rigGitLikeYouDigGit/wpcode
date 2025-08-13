

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
class DirectionXPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : CurveIntersect = None
	pass
class DirectionYPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : CurveIntersect = None
	pass
class DirectionZPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : CurveIntersect = None
	pass
class DirectionPlug(Plug):
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	dx_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	dy_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	dz_ : DirectionZPlug = PlugDescriptor("directionZ")
	node : CurveIntersect = None
	pass
class InputCurve1Plug(Plug):
	node : CurveIntersect = None
	pass
class InputCurve2Plug(Plug):
	node : CurveIntersect = None
	pass
class Parameter1Plug(Plug):
	node : CurveIntersect = None
	pass
class Parameter2Plug(Plug):
	node : CurveIntersect = None
	pass
class TolerancePlug(Plug):
	node : CurveIntersect = None
	pass
class UseDirectionPlug(Plug):
	node : CurveIntersect = None
	pass
# endregion


# define node class
class CurveIntersect(AbstractBaseCreate):
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	direction_ : DirectionPlug = PlugDescriptor("direction")
	inputCurve1_ : InputCurve1Plug = PlugDescriptor("inputCurve1")
	inputCurve2_ : InputCurve2Plug = PlugDescriptor("inputCurve2")
	parameter1_ : Parameter1Plug = PlugDescriptor("parameter1")
	parameter2_ : Parameter2Plug = PlugDescriptor("parameter2")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")
	useDirection_ : UseDirectionPlug = PlugDescriptor("useDirection")

	# node attributes

	typeName = "curveIntersect"
	typeIdInt = 1313030985
	pass

