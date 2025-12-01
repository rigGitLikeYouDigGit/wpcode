

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
class DistancePlug(Plug):
	node : DistanceDimShape = None
	pass
class EndPointXPlug(Plug):
	parent : EndPointPlug = PlugDescriptor("endPoint")
	node : DistanceDimShape = None
	pass
class EndPointYPlug(Plug):
	parent : EndPointPlug = PlugDescriptor("endPoint")
	node : DistanceDimShape = None
	pass
class EndPointZPlug(Plug):
	parent : EndPointPlug = PlugDescriptor("endPoint")
	node : DistanceDimShape = None
	pass
class EndPointPlug(Plug):
	endPointX_ : EndPointXPlug = PlugDescriptor("endPointX")
	epx_ : EndPointXPlug = PlugDescriptor("endPointX")
	endPointY_ : EndPointYPlug = PlugDescriptor("endPointY")
	epy_ : EndPointYPlug = PlugDescriptor("endPointY")
	endPointZ_ : EndPointZPlug = PlugDescriptor("endPointZ")
	epz_ : EndPointZPlug = PlugDescriptor("endPointZ")
	node : DistanceDimShape = None
	pass
class PrecisionPlug(Plug):
	node : DistanceDimShape = None
	pass
class StartPointXPlug(Plug):
	parent : StartPointPlug = PlugDescriptor("startPoint")
	node : DistanceDimShape = None
	pass
class StartPointYPlug(Plug):
	parent : StartPointPlug = PlugDescriptor("startPoint")
	node : DistanceDimShape = None
	pass
class StartPointZPlug(Plug):
	parent : StartPointPlug = PlugDescriptor("startPoint")
	node : DistanceDimShape = None
	pass
class StartPointPlug(Plug):
	startPointX_ : StartPointXPlug = PlugDescriptor("startPointX")
	spx_ : StartPointXPlug = PlugDescriptor("startPointX")
	startPointY_ : StartPointYPlug = PlugDescriptor("startPointY")
	spy_ : StartPointYPlug = PlugDescriptor("startPointY")
	startPointZ_ : StartPointZPlug = PlugDescriptor("startPointZ")
	spz_ : StartPointZPlug = PlugDescriptor("startPointZ")
	node : DistanceDimShape = None
	pass
# endregion


# define node class
class DistanceDimShape(DimensionShape):
	distance_ : DistancePlug = PlugDescriptor("distance")
	endPointX_ : EndPointXPlug = PlugDescriptor("endPointX")
	endPointY_ : EndPointYPlug = PlugDescriptor("endPointY")
	endPointZ_ : EndPointZPlug = PlugDescriptor("endPointZ")
	endPoint_ : EndPointPlug = PlugDescriptor("endPoint")
	precision_ : PrecisionPlug = PlugDescriptor("precision")
	startPointX_ : StartPointXPlug = PlugDescriptor("startPointX")
	startPointY_ : StartPointYPlug = PlugDescriptor("startPointY")
	startPointZ_ : StartPointZPlug = PlugDescriptor("startPointZ")
	startPoint_ : StartPointPlug = PlugDescriptor("startPoint")

	# node attributes

	typeName = "distanceDimShape"
	typeIdInt = 1145326926
	nodeLeafClassAttrs = ["distance", "endPointX", "endPointY", "endPointZ", "endPoint", "precision", "startPointX", "startPointY", "startPointZ", "startPoint"]
	nodeLeafPlugs = ["distance", "endPoint", "precision", "startPoint"]
	pass

