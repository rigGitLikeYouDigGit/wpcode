

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
class NormalXPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : AvgNurbsSurfacePoints = None
	pass
class NormalYPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : AvgNurbsSurfacePoints = None
	pass
class NormalZPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : AvgNurbsSurfacePoints = None
	pass
class NormalPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	nx_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	ny_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	nz_ : NormalZPlug = PlugDescriptor("normalZ")
	node : AvgNurbsSurfacePoints = None
	pass
class PositionXPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : AvgNurbsSurfacePoints = None
	pass
class PositionYPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : AvgNurbsSurfacePoints = None
	pass
class PositionZPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : AvgNurbsSurfacePoints = None
	pass
class PositionPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	px_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	py_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	pz_ : PositionZPlug = PlugDescriptor("positionZ")
	node : AvgNurbsSurfacePoints = None
	pass
class ResultPlug(Plug):
	normal_ : NormalPlug = PlugDescriptor("normal")
	n_ : NormalPlug = PlugDescriptor("normal")
	position_ : PositionPlug = PlugDescriptor("position")
	p_ : PositionPlug = PlugDescriptor("position")
	node : AvgNurbsSurfacePoints = None
	pass
class CvIthIndexPlug(Plug):
	parent : SurfacePointPlug = PlugDescriptor("surfacePoint")
	node : AvgNurbsSurfacePoints = None
	pass
class CvJthIndexPlug(Plug):
	parent : SurfacePointPlug = PlugDescriptor("surfacePoint")
	node : AvgNurbsSurfacePoints = None
	pass
class InputSurfacePlug(Plug):
	parent : SurfacePointPlug = PlugDescriptor("surfacePoint")
	node : AvgNurbsSurfacePoints = None
	pass
class ParameterUPlug(Plug):
	parent : SurfacePointPlug = PlugDescriptor("surfacePoint")
	node : AvgNurbsSurfacePoints = None
	pass
class ParameterVPlug(Plug):
	parent : SurfacePointPlug = PlugDescriptor("surfacePoint")
	node : AvgNurbsSurfacePoints = None
	pass
class WeightPlug(Plug):
	parent : SurfacePointPlug = PlugDescriptor("surfacePoint")
	node : AvgNurbsSurfacePoints = None
	pass
class SurfacePointPlug(Plug):
	cvIthIndex_ : CvIthIndexPlug = PlugDescriptor("cvIthIndex")
	ci_ : CvIthIndexPlug = PlugDescriptor("cvIthIndex")
	cvJthIndex_ : CvJthIndexPlug = PlugDescriptor("cvJthIndex")
	cj_ : CvJthIndexPlug = PlugDescriptor("cvJthIndex")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	is_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	parameterU_ : ParameterUPlug = PlugDescriptor("parameterU")
	u_ : ParameterUPlug = PlugDescriptor("parameterU")
	parameterV_ : ParameterVPlug = PlugDescriptor("parameterV")
	v_ : ParameterVPlug = PlugDescriptor("parameterV")
	weight_ : WeightPlug = PlugDescriptor("weight")
	wt_ : WeightPlug = PlugDescriptor("weight")
	node : AvgNurbsSurfacePoints = None
	pass
# endregion


# define node class
class AvgNurbsSurfacePoints(AbstractBaseCreate):
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	normal_ : NormalPlug = PlugDescriptor("normal")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	position_ : PositionPlug = PlugDescriptor("position")
	result_ : ResultPlug = PlugDescriptor("result")
	cvIthIndex_ : CvIthIndexPlug = PlugDescriptor("cvIthIndex")
	cvJthIndex_ : CvJthIndexPlug = PlugDescriptor("cvJthIndex")
	inputSurface_ : InputSurfacePlug = PlugDescriptor("inputSurface")
	parameterU_ : ParameterUPlug = PlugDescriptor("parameterU")
	parameterV_ : ParameterVPlug = PlugDescriptor("parameterV")
	weight_ : WeightPlug = PlugDescriptor("weight")
	surfacePoint_ : SurfacePointPlug = PlugDescriptor("surfacePoint")

	# node attributes

	typeName = "avgNurbsSurfacePoints"
	apiTypeInt = 47
	apiTypeStr = "kAvgNurbsSurfacePoints"
	typeIdInt = 1312902736
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["normalX", "normalY", "normalZ", "normal", "positionX", "positionY", "positionZ", "position", "result", "cvIthIndex", "cvJthIndex", "inputSurface", "parameterU", "parameterV", "weight", "surfacePoint"]
	nodeLeafPlugs = ["result", "surfacePoint"]
	pass

