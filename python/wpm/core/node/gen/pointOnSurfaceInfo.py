

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
class InputSurfacePlug(Plug):
	node : PointOnSurfaceInfo = None
	pass
class ParameterUPlug(Plug):
	node : PointOnSurfaceInfo = None
	pass
class ParameterVPlug(Plug):
	node : PointOnSurfaceInfo = None
	pass
class NormalXPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : PointOnSurfaceInfo = None
	pass
class NormalYPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : PointOnSurfaceInfo = None
	pass
class NormalZPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : PointOnSurfaceInfo = None
	pass
class NormalPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	nx_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	ny_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	nz_ : NormalZPlug = PlugDescriptor("normalZ")
	node : PointOnSurfaceInfo = None
	pass
class NormalizedNormalXPlug(Plug):
	parent : NormalizedNormalPlug = PlugDescriptor("normalizedNormal")
	node : PointOnSurfaceInfo = None
	pass
class NormalizedNormalYPlug(Plug):
	parent : NormalizedNormalPlug = PlugDescriptor("normalizedNormal")
	node : PointOnSurfaceInfo = None
	pass
class NormalizedNormalZPlug(Plug):
	parent : NormalizedNormalPlug = PlugDescriptor("normalizedNormal")
	node : PointOnSurfaceInfo = None
	pass
class NormalizedNormalPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	normalizedNormalX_ : NormalizedNormalXPlug = PlugDescriptor("normalizedNormalX")
	nnx_ : NormalizedNormalXPlug = PlugDescriptor("normalizedNormalX")
	normalizedNormalY_ : NormalizedNormalYPlug = PlugDescriptor("normalizedNormalY")
	nny_ : NormalizedNormalYPlug = PlugDescriptor("normalizedNormalY")
	normalizedNormalZ_ : NormalizedNormalZPlug = PlugDescriptor("normalizedNormalZ")
	nnz_ : NormalizedNormalZPlug = PlugDescriptor("normalizedNormalZ")
	node : PointOnSurfaceInfo = None
	pass
class NormalizedTangentUXPlug(Plug):
	parent : NormalizedTangentUPlug = PlugDescriptor("normalizedTangentU")
	node : PointOnSurfaceInfo = None
	pass
class NormalizedTangentUYPlug(Plug):
	parent : NormalizedTangentUPlug = PlugDescriptor("normalizedTangentU")
	node : PointOnSurfaceInfo = None
	pass
class NormalizedTangentUZPlug(Plug):
	parent : NormalizedTangentUPlug = PlugDescriptor("normalizedTangentU")
	node : PointOnSurfaceInfo = None
	pass
class NormalizedTangentUPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	normalizedTangentUX_ : NormalizedTangentUXPlug = PlugDescriptor("normalizedTangentUX")
	nux_ : NormalizedTangentUXPlug = PlugDescriptor("normalizedTangentUX")
	normalizedTangentUY_ : NormalizedTangentUYPlug = PlugDescriptor("normalizedTangentUY")
	nuy_ : NormalizedTangentUYPlug = PlugDescriptor("normalizedTangentUY")
	normalizedTangentUZ_ : NormalizedTangentUZPlug = PlugDescriptor("normalizedTangentUZ")
	nuz_ : NormalizedTangentUZPlug = PlugDescriptor("normalizedTangentUZ")
	node : PointOnSurfaceInfo = None
	pass
class NormalizedTangentVXPlug(Plug):
	parent : NormalizedTangentVPlug = PlugDescriptor("normalizedTangentV")
	node : PointOnSurfaceInfo = None
	pass
class NormalizedTangentVYPlug(Plug):
	parent : NormalizedTangentVPlug = PlugDescriptor("normalizedTangentV")
	node : PointOnSurfaceInfo = None
	pass
class NormalizedTangentVZPlug(Plug):
	parent : NormalizedTangentVPlug = PlugDescriptor("normalizedTangentV")
	node : PointOnSurfaceInfo = None
	pass
class NormalizedTangentVPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	normalizedTangentVX_ : NormalizedTangentVXPlug = PlugDescriptor("normalizedTangentVX")
	nvx_ : NormalizedTangentVXPlug = PlugDescriptor("normalizedTangentVX")
	normalizedTangentVY_ : NormalizedTangentVYPlug = PlugDescriptor("normalizedTangentVY")
	nvy_ : NormalizedTangentVYPlug = PlugDescriptor("normalizedTangentVY")
	normalizedTangentVZ_ : NormalizedTangentVZPlug = PlugDescriptor("normalizedTangentVZ")
	nvz_ : NormalizedTangentVZPlug = PlugDescriptor("normalizedTangentVZ")
	node : PointOnSurfaceInfo = None
	pass
class PositionXPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : PointOnSurfaceInfo = None
	pass
class PositionYPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : PointOnSurfaceInfo = None
	pass
class PositionZPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : PointOnSurfaceInfo = None
	pass
class PositionPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	px_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	py_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	pz_ : PositionZPlug = PlugDescriptor("positionZ")
	node : PointOnSurfaceInfo = None
	pass
class TangentUxPlug(Plug):
	parent : TangentUPlug = PlugDescriptor("tangentU")
	node : PointOnSurfaceInfo = None
	pass
class TangentUyPlug(Plug):
	parent : TangentUPlug = PlugDescriptor("tangentU")
	node : PointOnSurfaceInfo = None
	pass
class TangentUzPlug(Plug):
	parent : TangentUPlug = PlugDescriptor("tangentU")
	node : PointOnSurfaceInfo = None
	pass
class TangentUPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	tangentUx_ : TangentUxPlug = PlugDescriptor("tangentUx")
	tux_ : TangentUxPlug = PlugDescriptor("tangentUx")
	tangentUy_ : TangentUyPlug = PlugDescriptor("tangentUy")
	tuy_ : TangentUyPlug = PlugDescriptor("tangentUy")
	tangentUz_ : TangentUzPlug = PlugDescriptor("tangentUz")
	tuz_ : TangentUzPlug = PlugDescriptor("tangentUz")
	node : PointOnSurfaceInfo = None
	pass
class TangentVxPlug(Plug):
	parent : TangentVPlug = PlugDescriptor("tangentV")
	node : PointOnSurfaceInfo = None
	pass
class TangentVyPlug(Plug):
	parent : TangentVPlug = PlugDescriptor("tangentV")
	node : PointOnSurfaceInfo = None
	pass
class TangentVzPlug(Plug):
	parent : TangentVPlug = PlugDescriptor("tangentV")
	node : PointOnSurfaceInfo = None
	pass
class TangentVPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	tangentVx_ : TangentVxPlug = PlugDescriptor("tangentVx")
	tvx_ : TangentVxPlug = PlugDescriptor("tangentVx")
	tangentVy_ : TangentVyPlug = PlugDescriptor("tangentVy")
	tvy_ : TangentVyPlug = PlugDescriptor("tangentVy")
	tangentVz_ : TangentVzPlug = PlugDescriptor("tangentVz")
	tvz_ : TangentVzPlug = PlugDescriptor("tangentVz")
	node : PointOnSurfaceInfo = None
	pass
class ResultPlug(Plug):
	normal_ : NormalPlug = PlugDescriptor("normal")
	n_ : NormalPlug = PlugDescriptor("normal")
	normalizedNormal_ : NormalizedNormalPlug = PlugDescriptor("normalizedNormal")
	nn_ : NormalizedNormalPlug = PlugDescriptor("normalizedNormal")
	normalizedTangentU_ : NormalizedTangentUPlug = PlugDescriptor("normalizedTangentU")
	ntu_ : NormalizedTangentUPlug = PlugDescriptor("normalizedTangentU")
	normalizedTangentV_ : NormalizedTangentVPlug = PlugDescriptor("normalizedTangentV")
	ntv_ : NormalizedTangentVPlug = PlugDescriptor("normalizedTangentV")
	position_ : PositionPlug = PlugDescriptor("position")
	p_ : PositionPlug = PlugDescriptor("position")
	tangentU_ : TangentUPlug = PlugDescriptor("tangentU")
	tu_ : TangentUPlug = PlugDescriptor("tangentU")
	tangentV_ : TangentVPlug = PlugDescriptor("tangentV")
	tv_ : TangentVPlug = PlugDescriptor("tangentV")
	node : PointOnSurfaceInfo = None
	pass
class TurnOnPercentagePlug(Plug):
	node : PointOnSurfaceInfo = None
	pass
# endregion


# define node class
class PointOnSurfaceInfo(AbstractBaseCreate):
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	parameterU_ : ParameterUPlug = PlugDescriptor("parameterU")
	parameterV_ : ParameterVPlug = PlugDescriptor("parameterV")
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	normal_ : NormalPlug = PlugDescriptor("normal")
	normalizedNormalX_ : NormalizedNormalXPlug = PlugDescriptor("normalizedNormalX")
	normalizedNormalY_ : NormalizedNormalYPlug = PlugDescriptor("normalizedNormalY")
	normalizedNormalZ_ : NormalizedNormalZPlug = PlugDescriptor("normalizedNormalZ")
	normalizedNormal_ : NormalizedNormalPlug = PlugDescriptor("normalizedNormal")
	normalizedTangentUX_ : NormalizedTangentUXPlug = PlugDescriptor("normalizedTangentUX")
	normalizedTangentUY_ : NormalizedTangentUYPlug = PlugDescriptor("normalizedTangentUY")
	normalizedTangentUZ_ : NormalizedTangentUZPlug = PlugDescriptor("normalizedTangentUZ")
	normalizedTangentU_ : NormalizedTangentUPlug = PlugDescriptor("normalizedTangentU")
	normalizedTangentVX_ : NormalizedTangentVXPlug = PlugDescriptor("normalizedTangentVX")
	normalizedTangentVY_ : NormalizedTangentVYPlug = PlugDescriptor("normalizedTangentVY")
	normalizedTangentVZ_ : NormalizedTangentVZPlug = PlugDescriptor("normalizedTangentVZ")
	normalizedTangentV_ : NormalizedTangentVPlug = PlugDescriptor("normalizedTangentV")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	position_ : PositionPlug = PlugDescriptor("position")
	tangentUx_ : TangentUxPlug = PlugDescriptor("tangentUx")
	tangentUy_ : TangentUyPlug = PlugDescriptor("tangentUy")
	tangentUz_ : TangentUzPlug = PlugDescriptor("tangentUz")
	tangentU_ : TangentUPlug = PlugDescriptor("tangentU")
	tangentVx_ : TangentVxPlug = PlugDescriptor("tangentVx")
	tangentVy_ : TangentVyPlug = PlugDescriptor("tangentVy")
	tangentVz_ : TangentVzPlug = PlugDescriptor("tangentVz")
	tangentV_ : TangentVPlug = PlugDescriptor("tangentV")
	result_ : ResultPlug = PlugDescriptor("result")
	turnOnPercentage_ : TurnOnPercentagePlug = PlugDescriptor("turnOnPercentage")

	# node attributes

	typeName = "pointOnSurfaceInfo"
	apiTypeInt = 85
	apiTypeStr = "kPointOnSurfaceInfo"
	typeIdInt = 1313887049
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["inputSurface", "parameterU", "parameterV", "normalX", "normalY", "normalZ", "normal", "normalizedNormalX", "normalizedNormalY", "normalizedNormalZ", "normalizedNormal", "normalizedTangentUX", "normalizedTangentUY", "normalizedTangentUZ", "normalizedTangentU", "normalizedTangentVX", "normalizedTangentVY", "normalizedTangentVZ", "normalizedTangentV", "positionX", "positionY", "positionZ", "position", "tangentUx", "tangentUy", "tangentUz", "tangentU", "tangentVx", "tangentVy", "tangentVz", "tangentV", "result", "turnOnPercentage"]
	nodeLeafPlugs = ["inputSurface", "parameterU", "parameterV", "result", "turnOnPercentage"]
	pass

