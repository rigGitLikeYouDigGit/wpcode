

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
DimensionShape = retriever.getNodeCls("DimensionShape")
assert DimensionShape
if T.TYPE_CHECKING:
	from .. import DimensionShape

# add node doc



# region plug type defs
class AnglePlug(Plug):
	node : AngleDimension = None
	pass
class EndPointXPlug(Plug):
	parent : EndPointPlug = PlugDescriptor("endPoint")
	node : AngleDimension = None
	pass
class EndPointYPlug(Plug):
	parent : EndPointPlug = PlugDescriptor("endPoint")
	node : AngleDimension = None
	pass
class EndPointZPlug(Plug):
	parent : EndPointPlug = PlugDescriptor("endPoint")
	node : AngleDimension = None
	pass
class EndPointPlug(Plug):
	endPointX_ : EndPointXPlug = PlugDescriptor("endPointX")
	epx_ : EndPointXPlug = PlugDescriptor("endPointX")
	endPointY_ : EndPointYPlug = PlugDescriptor("endPointY")
	epy_ : EndPointYPlug = PlugDescriptor("endPointY")
	endPointZ_ : EndPointZPlug = PlugDescriptor("endPointZ")
	epz_ : EndPointZPlug = PlugDescriptor("endPointZ")
	node : AngleDimension = None
	pass
class MiddlePointXPlug(Plug):
	parent : MiddlePointPlug = PlugDescriptor("middlePoint")
	node : AngleDimension = None
	pass
class MiddlePointYPlug(Plug):
	parent : MiddlePointPlug = PlugDescriptor("middlePoint")
	node : AngleDimension = None
	pass
class MiddlePointZPlug(Plug):
	parent : MiddlePointPlug = PlugDescriptor("middlePoint")
	node : AngleDimension = None
	pass
class MiddlePointPlug(Plug):
	middlePointX_ : MiddlePointXPlug = PlugDescriptor("middlePointX")
	mpx_ : MiddlePointXPlug = PlugDescriptor("middlePointX")
	middlePointY_ : MiddlePointYPlug = PlugDescriptor("middlePointY")
	mpy_ : MiddlePointYPlug = PlugDescriptor("middlePointY")
	middlePointZ_ : MiddlePointZPlug = PlugDescriptor("middlePointZ")
	mpz_ : MiddlePointZPlug = PlugDescriptor("middlePointZ")
	node : AngleDimension = None
	pass
class StartPointXPlug(Plug):
	parent : StartPointPlug = PlugDescriptor("startPoint")
	node : AngleDimension = None
	pass
class StartPointYPlug(Plug):
	parent : StartPointPlug = PlugDescriptor("startPoint")
	node : AngleDimension = None
	pass
class StartPointZPlug(Plug):
	parent : StartPointPlug = PlugDescriptor("startPoint")
	node : AngleDimension = None
	pass
class StartPointPlug(Plug):
	startPointX_ : StartPointXPlug = PlugDescriptor("startPointX")
	spx_ : StartPointXPlug = PlugDescriptor("startPointX")
	startPointY_ : StartPointYPlug = PlugDescriptor("startPointY")
	spy_ : StartPointYPlug = PlugDescriptor("startPointY")
	startPointZ_ : StartPointZPlug = PlugDescriptor("startPointZ")
	spz_ : StartPointZPlug = PlugDescriptor("startPointZ")
	node : AngleDimension = None
	pass
# endregion


# define node class
class AngleDimension(DimensionShape):
	angle_ : AnglePlug = PlugDescriptor("angle")
	endPointX_ : EndPointXPlug = PlugDescriptor("endPointX")
	endPointY_ : EndPointYPlug = PlugDescriptor("endPointY")
	endPointZ_ : EndPointZPlug = PlugDescriptor("endPointZ")
	endPoint_ : EndPointPlug = PlugDescriptor("endPoint")
	middlePointX_ : MiddlePointXPlug = PlugDescriptor("middlePointX")
	middlePointY_ : MiddlePointYPlug = PlugDescriptor("middlePointY")
	middlePointZ_ : MiddlePointZPlug = PlugDescriptor("middlePointZ")
	middlePoint_ : MiddlePointPlug = PlugDescriptor("middlePoint")
	startPointX_ : StartPointXPlug = PlugDescriptor("startPointX")
	startPointY_ : StartPointYPlug = PlugDescriptor("startPointY")
	startPointZ_ : StartPointZPlug = PlugDescriptor("startPointZ")
	startPoint_ : StartPointPlug = PlugDescriptor("startPoint")

	# node attributes

	typeName = "angleDimension"
	typeIdInt = 1095189582
	nodeLeafClassAttrs = ["angle", "endPointX", "endPointY", "endPointZ", "endPoint", "middlePointX", "middlePointY", "middlePointZ", "middlePoint", "startPointX", "startPointY", "startPointZ", "startPoint"]
	nodeLeafPlugs = ["angle", "endPoint", "middlePoint", "startPoint"]
	pass

