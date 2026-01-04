

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
class InputSurfacesPlug(Plug):
	node : AvgSurfacePoints = None
	pass
class ParameterUPlug(Plug):
	node : AvgSurfacePoints = None
	pass
class ParameterVPlug(Plug):
	node : AvgSurfacePoints = None
	pass
class NormalXPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : AvgSurfacePoints = None
	pass
class NormalYPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : AvgSurfacePoints = None
	pass
class NormalZPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : AvgSurfacePoints = None
	pass
class NormalPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	nx_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	ny_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	nz_ : NormalZPlug = PlugDescriptor("normalZ")
	node : AvgSurfacePoints = None
	pass
class PositionXPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : AvgSurfacePoints = None
	pass
class PositionYPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : AvgSurfacePoints = None
	pass
class PositionZPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : AvgSurfacePoints = None
	pass
class PositionPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	px_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	py_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	pz_ : PositionZPlug = PlugDescriptor("positionZ")
	node : AvgSurfacePoints = None
	pass
class ResultPlug(Plug):
	normal_ : NormalPlug = PlugDescriptor("normal")
	n_ : NormalPlug = PlugDescriptor("normal")
	position_ : PositionPlug = PlugDescriptor("position")
	p_ : PositionPlug = PlugDescriptor("position")
	node : AvgSurfacePoints = None
	pass
class TurnOnPercentagePlug(Plug):
	node : AvgSurfacePoints = None
	pass
class WeightPlug(Plug):
	node : AvgSurfacePoints = None
	pass
# endregion


# define node class
class AvgSurfacePoints(AbstractBaseCreate):
	inputSurfaces_ : InputSurfacesPlug = PlugDescriptor("inputSurfaces")
	parameterU_ : ParameterUPlug = PlugDescriptor("parameterU")
	parameterV_ : ParameterVPlug = PlugDescriptor("parameterV")
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	normal_ : NormalPlug = PlugDescriptor("normal")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	position_ : PositionPlug = PlugDescriptor("position")
	result_ : ResultPlug = PlugDescriptor("result")
	turnOnPercentage_ : TurnOnPercentagePlug = PlugDescriptor("turnOnPercentage")
	weight_ : WeightPlug = PlugDescriptor("weight")

	# node attributes

	typeName = "avgSurfacePoints"
	apiTypeInt = 46
	apiTypeStr = "kAvgSurfacePoints"
	typeIdInt = 1312904016
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["inputSurfaces", "parameterU", "parameterV", "normalX", "normalY", "normalZ", "normal", "positionX", "positionY", "positionZ", "position", "result", "turnOnPercentage", "weight"]
	nodeLeafPlugs = ["inputSurfaces", "parameterU", "parameterV", "result", "turnOnPercentage", "weight"]
	pass

