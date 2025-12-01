

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
class AutomaticPlug(Plug):
	node : PolyProjectCurve = None
	pass
class TrianglePlug(Plug):
	parent : ProjectedPointPlug = PlugDescriptor("projectedPoint")
	node : PolyProjectCurve = None
	pass
class ProjectedPointPlug(Plug):
	parent : CurvePointsPlug = PlugDescriptor("curvePoints")
	baryCoord_ : BaryCoordPlug = PlugDescriptor("baryCoord")
	bc_ : BaryCoordPlug = PlugDescriptor("baryCoord")
	face_ : FacePlug = PlugDescriptor("face")
	f_ : FacePlug = PlugDescriptor("face")
	triangle_ : TrianglePlug = PlugDescriptor("triangle")
	t_ : TrianglePlug = PlugDescriptor("triangle")
	node : PolyProjectCurve = None
	pass
class CurvePointsPlug(Plug):
	projectedPoint_ : ProjectedPointPlug = PlugDescriptor("projectedPoint")
	pp_ : ProjectedPointPlug = PlugDescriptor("projectedPoint")
	node : PolyProjectCurve = None
	pass
class CurveSamplesPlug(Plug):
	node : PolyProjectCurve = None
	pass
class DirectionXPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : PolyProjectCurve = None
	pass
class DirectionYPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : PolyProjectCurve = None
	pass
class DirectionZPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : PolyProjectCurve = None
	pass
class DirectionPlug(Plug):
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	dx_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	dy_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	dz_ : DirectionZPlug = PlugDescriptor("directionZ")
	node : PolyProjectCurve = None
	pass
class InputCurvePlug(Plug):
	node : PolyProjectCurve = None
	pass
class InputMatrixPlug(Plug):
	node : PolyProjectCurve = None
	pass
class InputMeshPlug(Plug):
	node : PolyProjectCurve = None
	pass
class OutputCurvePlug(Plug):
	node : PolyProjectCurve = None
	pass
class PointsOnEdgesPlug(Plug):
	node : PolyProjectCurve = None
	pass
class BaryCoord1Plug(Plug):
	parent : BaryCoordPlug = PlugDescriptor("baryCoord")
	node : PolyProjectCurve = None
	pass
class BaryCoord2Plug(Plug):
	parent : BaryCoordPlug = PlugDescriptor("baryCoord")
	node : PolyProjectCurve = None
	pass
class BaryCoord3Plug(Plug):
	parent : BaryCoordPlug = PlugDescriptor("baryCoord")
	node : PolyProjectCurve = None
	pass
class BaryCoordPlug(Plug):
	parent : ProjectedPointPlug = PlugDescriptor("projectedPoint")
	baryCoord1_ : BaryCoord1Plug = PlugDescriptor("baryCoord1")
	bc1_ : BaryCoord1Plug = PlugDescriptor("baryCoord1")
	baryCoord2_ : BaryCoord2Plug = PlugDescriptor("baryCoord2")
	bc2_ : BaryCoord2Plug = PlugDescriptor("baryCoord2")
	baryCoord3_ : BaryCoord3Plug = PlugDescriptor("baryCoord3")
	bc3_ : BaryCoord3Plug = PlugDescriptor("baryCoord3")
	node : PolyProjectCurve = None
	pass
class FacePlug(Plug):
	parent : ProjectedPointPlug = PlugDescriptor("projectedPoint")
	node : PolyProjectCurve = None
	pass
class TolerancePlug(Plug):
	node : PolyProjectCurve = None
	pass
# endregion


# define node class
class PolyProjectCurve(AbstractBaseCreate):
	automatic_ : AutomaticPlug = PlugDescriptor("automatic")
	triangle_ : TrianglePlug = PlugDescriptor("triangle")
	projectedPoint_ : ProjectedPointPlug = PlugDescriptor("projectedPoint")
	curvePoints_ : CurvePointsPlug = PlugDescriptor("curvePoints")
	curveSamples_ : CurveSamplesPlug = PlugDescriptor("curveSamples")
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	direction_ : DirectionPlug = PlugDescriptor("direction")
	inputCurve_ : InputCurvePlug = PlugDescriptor("inputCurve")
	inputMatrix_ : InputMatrixPlug = PlugDescriptor("inputMatrix")
	inputMesh_ : InputMeshPlug = PlugDescriptor("inputMesh")
	outputCurve_ : OutputCurvePlug = PlugDescriptor("outputCurve")
	pointsOnEdges_ : PointsOnEdgesPlug = PlugDescriptor("pointsOnEdges")
	baryCoord1_ : BaryCoord1Plug = PlugDescriptor("baryCoord1")
	baryCoord2_ : BaryCoord2Plug = PlugDescriptor("baryCoord2")
	baryCoord3_ : BaryCoord3Plug = PlugDescriptor("baryCoord3")
	baryCoord_ : BaryCoordPlug = PlugDescriptor("baryCoord")
	face_ : FacePlug = PlugDescriptor("face")
	tolerance_ : TolerancePlug = PlugDescriptor("tolerance")

	# node attributes

	typeName = "polyProjectCurve"
	apiTypeInt = 1072
	apiTypeStr = "kPolyProjectCurve"
	typeIdInt = 1347437398
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["automatic", "triangle", "projectedPoint", "curvePoints", "curveSamples", "directionX", "directionY", "directionZ", "direction", "inputCurve", "inputMatrix", "inputMesh", "outputCurve", "pointsOnEdges", "baryCoord1", "baryCoord2", "baryCoord3", "baryCoord", "face", "tolerance"]
	nodeLeafPlugs = ["automatic", "curvePoints", "curveSamples", "direction", "inputCurve", "inputMatrix", "inputMesh", "outputCurve", "pointsOnEdges", "tolerance"]
	pass

