

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
class InputCurvePlug(Plug):
	node : PointOnCurveInfo = None
	pass
class ParameterPlug(Plug):
	node : PointOnCurveInfo = None
	pass
class CurvatureCenterXPlug(Plug):
	parent : CurvatureCenterPlug = PlugDescriptor("curvatureCenter")
	node : PointOnCurveInfo = None
	pass
class CurvatureCenterYPlug(Plug):
	parent : CurvatureCenterPlug = PlugDescriptor("curvatureCenter")
	node : PointOnCurveInfo = None
	pass
class CurvatureCenterZPlug(Plug):
	parent : CurvatureCenterPlug = PlugDescriptor("curvatureCenter")
	node : PointOnCurveInfo = None
	pass
class CurvatureCenterPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	curvatureCenterX_ : CurvatureCenterXPlug = PlugDescriptor("curvatureCenterX")
	ccx_ : CurvatureCenterXPlug = PlugDescriptor("curvatureCenterX")
	curvatureCenterY_ : CurvatureCenterYPlug = PlugDescriptor("curvatureCenterY")
	ccy_ : CurvatureCenterYPlug = PlugDescriptor("curvatureCenterY")
	curvatureCenterZ_ : CurvatureCenterZPlug = PlugDescriptor("curvatureCenterZ")
	ccz_ : CurvatureCenterZPlug = PlugDescriptor("curvatureCenterZ")
	node : PointOnCurveInfo = None
	pass
class CurvatureRadiusPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	node : PointOnCurveInfo = None
	pass
class NormalXPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : PointOnCurveInfo = None
	pass
class NormalYPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : PointOnCurveInfo = None
	pass
class NormalZPlug(Plug):
	parent : NormalPlug = PlugDescriptor("normal")
	node : PointOnCurveInfo = None
	pass
class NormalPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	nx_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	ny_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	nz_ : NormalZPlug = PlugDescriptor("normalZ")
	node : PointOnCurveInfo = None
	pass
class NormalizedNormalXPlug(Plug):
	parent : NormalizedNormalPlug = PlugDescriptor("normalizedNormal")
	node : PointOnCurveInfo = None
	pass
class NormalizedNormalYPlug(Plug):
	parent : NormalizedNormalPlug = PlugDescriptor("normalizedNormal")
	node : PointOnCurveInfo = None
	pass
class NormalizedNormalZPlug(Plug):
	parent : NormalizedNormalPlug = PlugDescriptor("normalizedNormal")
	node : PointOnCurveInfo = None
	pass
class NormalizedNormalPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	normalizedNormalX_ : NormalizedNormalXPlug = PlugDescriptor("normalizedNormalX")
	nnx_ : NormalizedNormalXPlug = PlugDescriptor("normalizedNormalX")
	normalizedNormalY_ : NormalizedNormalYPlug = PlugDescriptor("normalizedNormalY")
	nny_ : NormalizedNormalYPlug = PlugDescriptor("normalizedNormalY")
	normalizedNormalZ_ : NormalizedNormalZPlug = PlugDescriptor("normalizedNormalZ")
	nnz_ : NormalizedNormalZPlug = PlugDescriptor("normalizedNormalZ")
	node : PointOnCurveInfo = None
	pass
class NormalizedTangentXPlug(Plug):
	parent : NormalizedTangentPlug = PlugDescriptor("normalizedTangent")
	node : PointOnCurveInfo = None
	pass
class NormalizedTangentYPlug(Plug):
	parent : NormalizedTangentPlug = PlugDescriptor("normalizedTangent")
	node : PointOnCurveInfo = None
	pass
class NormalizedTangentZPlug(Plug):
	parent : NormalizedTangentPlug = PlugDescriptor("normalizedTangent")
	node : PointOnCurveInfo = None
	pass
class NormalizedTangentPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	normalizedTangentX_ : NormalizedTangentXPlug = PlugDescriptor("normalizedTangentX")
	ntx_ : NormalizedTangentXPlug = PlugDescriptor("normalizedTangentX")
	normalizedTangentY_ : NormalizedTangentYPlug = PlugDescriptor("normalizedTangentY")
	nty_ : NormalizedTangentYPlug = PlugDescriptor("normalizedTangentY")
	normalizedTangentZ_ : NormalizedTangentZPlug = PlugDescriptor("normalizedTangentZ")
	ntz_ : NormalizedTangentZPlug = PlugDescriptor("normalizedTangentZ")
	node : PointOnCurveInfo = None
	pass
class PositionXPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : PointOnCurveInfo = None
	pass
class PositionYPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : PointOnCurveInfo = None
	pass
class PositionZPlug(Plug):
	parent : PositionPlug = PlugDescriptor("position")
	node : PointOnCurveInfo = None
	pass
class PositionPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	px_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	py_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	pz_ : PositionZPlug = PlugDescriptor("positionZ")
	node : PointOnCurveInfo = None
	pass
class TangentXPlug(Plug):
	parent : TangentPlug = PlugDescriptor("tangent")
	node : PointOnCurveInfo = None
	pass
class TangentYPlug(Plug):
	parent : TangentPlug = PlugDescriptor("tangent")
	node : PointOnCurveInfo = None
	pass
class TangentZPlug(Plug):
	parent : TangentPlug = PlugDescriptor("tangent")
	node : PointOnCurveInfo = None
	pass
class TangentPlug(Plug):
	parent : ResultPlug = PlugDescriptor("result")
	tangentX_ : TangentXPlug = PlugDescriptor("tangentX")
	tx_ : TangentXPlug = PlugDescriptor("tangentX")
	tangentY_ : TangentYPlug = PlugDescriptor("tangentY")
	ty_ : TangentYPlug = PlugDescriptor("tangentY")
	tangentZ_ : TangentZPlug = PlugDescriptor("tangentZ")
	tz_ : TangentZPlug = PlugDescriptor("tangentZ")
	node : PointOnCurveInfo = None
	pass
class ResultPlug(Plug):
	curvatureCenter_ : CurvatureCenterPlug = PlugDescriptor("curvatureCenter")
	cc_ : CurvatureCenterPlug = PlugDescriptor("curvatureCenter")
	curvatureRadius_ : CurvatureRadiusPlug = PlugDescriptor("curvatureRadius")
	cr_ : CurvatureRadiusPlug = PlugDescriptor("curvatureRadius")
	normal_ : NormalPlug = PlugDescriptor("normal")
	n_ : NormalPlug = PlugDescriptor("normal")
	normalizedNormal_ : NormalizedNormalPlug = PlugDescriptor("normalizedNormal")
	nn_ : NormalizedNormalPlug = PlugDescriptor("normalizedNormal")
	normalizedTangent_ : NormalizedTangentPlug = PlugDescriptor("normalizedTangent")
	nt_ : NormalizedTangentPlug = PlugDescriptor("normalizedTangent")
	position_ : PositionPlug = PlugDescriptor("position")
	p_ : PositionPlug = PlugDescriptor("position")
	tangent_ : TangentPlug = PlugDescriptor("tangent")
	t_ : TangentPlug = PlugDescriptor("tangent")
	node : PointOnCurveInfo = None
	pass
class TurnOnPercentagePlug(Plug):
	node : PointOnCurveInfo = None
	pass
# endregion


# define node class
class PointOnCurveInfo(AbstractBaseCreate):
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	parameter_ : ParameterPlug = PlugDescriptor("parameter")
	curvatureCenterX_ : CurvatureCenterXPlug = PlugDescriptor("curvatureCenterX")
	curvatureCenterY_ : CurvatureCenterYPlug = PlugDescriptor("curvatureCenterY")
	curvatureCenterZ_ : CurvatureCenterZPlug = PlugDescriptor("curvatureCenterZ")
	curvatureCenter_ : CurvatureCenterPlug = PlugDescriptor("curvatureCenter")
	curvatureRadius_ : CurvatureRadiusPlug = PlugDescriptor("curvatureRadius")
	normalX_ : NormalXPlug = PlugDescriptor("normalX")
	normalY_ : NormalYPlug = PlugDescriptor("normalY")
	normalZ_ : NormalZPlug = PlugDescriptor("normalZ")
	normal_ : NormalPlug = PlugDescriptor("normal")
	normalizedNormalX_ : NormalizedNormalXPlug = PlugDescriptor("normalizedNormalX")
	normalizedNormalY_ : NormalizedNormalYPlug = PlugDescriptor("normalizedNormalY")
	normalizedNormalZ_ : NormalizedNormalZPlug = PlugDescriptor("normalizedNormalZ")
	normalizedNormal_ : NormalizedNormalPlug = PlugDescriptor("normalizedNormal")
	normalizedTangentX_ : NormalizedTangentXPlug = PlugDescriptor("normalizedTangentX")
	normalizedTangentY_ : NormalizedTangentYPlug = PlugDescriptor("normalizedTangentY")
	normalizedTangentZ_ : NormalizedTangentZPlug = PlugDescriptor("normalizedTangentZ")
	normalizedTangent_ : NormalizedTangentPlug = PlugDescriptor("normalizedTangent")
	positionX_ : PositionXPlug = PlugDescriptor("positionX")
	positionY_ : PositionYPlug = PlugDescriptor("positionY")
	positionZ_ : PositionZPlug = PlugDescriptor("positionZ")
	position_ : PositionPlug = PlugDescriptor("position")
	tangentX_ : TangentXPlug = PlugDescriptor("tangentX")
	tangentY_ : TangentYPlug = PlugDescriptor("tangentY")
	tangentZ_ : TangentZPlug = PlugDescriptor("tangentZ")
	tangent_ : TangentPlug = PlugDescriptor("tangent")
	result_ : ResultPlug = PlugDescriptor("result")
	turnOnPercentage_ : TurnOnPercentagePlug = PlugDescriptor("turnOnPercentage")

	# node attributes

	typeName = "pointOnCurveInfo"
	apiTypeInt = 84
	apiTypeStr = "kPointOnCurveInfo"
	typeIdInt = 1313882953
	MFnCls = om.MFnDependencyNode
	pass

