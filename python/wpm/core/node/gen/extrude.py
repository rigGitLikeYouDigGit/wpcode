

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
class DegreeAlongLengthPlug(Plug):
	node : Extrude = None
	pass
class DirectionXPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : Extrude = None
	pass
class DirectionYPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : Extrude = None
	pass
class DirectionZPlug(Plug):
	parent : DirectionPlug = PlugDescriptor("direction")
	node : Extrude = None
	pass
class DirectionPlug(Plug):
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	dx_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	dy_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	dz_ : DirectionZPlug = PlugDescriptor("directionZ")
	node : Extrude = None
	pass
class ExtrudeTypePlug(Plug):
	node : Extrude = None
	pass
class FixedPathPlug(Plug):
	node : Extrude = None
	pass
class LengthPlug(Plug):
	node : Extrude = None
	pass
class OutputSurfacePlug(Plug):
	node : Extrude = None
	pass
class PathPlug(Plug):
	node : Extrude = None
	pass
class PivotXPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : Extrude = None
	pass
class PivotYPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : Extrude = None
	pass
class PivotZPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : Extrude = None
	pass
class PivotPlug(Plug):
	pivotX_ : PivotXPlug = PlugDescriptor("pivotX")
	px_ : PivotXPlug = PlugDescriptor("pivotX")
	pivotY_ : PivotYPlug = PlugDescriptor("pivotY")
	py_ : PivotYPlug = PlugDescriptor("pivotY")
	pivotZ_ : PivotZPlug = PlugDescriptor("pivotZ")
	pz_ : PivotZPlug = PlugDescriptor("pivotZ")
	node : Extrude = None
	pass
class ProfilePlug(Plug):
	node : Extrude = None
	pass
class ReverseSurfaceIfPathReversedPlug(Plug):
	node : Extrude = None
	pass
class RotationPlug(Plug):
	node : Extrude = None
	pass
class ScalePlug(Plug):
	node : Extrude = None
	pass
class SubCurveSubSurfacePlug(Plug):
	node : Extrude = None
	pass
class UseComponentPivotPlug(Plug):
	node : Extrude = None
	pass
class UseProfileNormalPlug(Plug):
	node : Extrude = None
	pass
# endregion


# define node class
class Extrude(AbstractBaseCreate):
	degreeAlongLength_ : DegreeAlongLengthPlug = PlugDescriptor("degreeAlongLength")
	directionX_ : DirectionXPlug = PlugDescriptor("directionX")
	directionY_ : DirectionYPlug = PlugDescriptor("directionY")
	directionZ_ : DirectionZPlug = PlugDescriptor("directionZ")
	direction_ : DirectionPlug = PlugDescriptor("direction")
	extrudeType_ : ExtrudeTypePlug = PlugDescriptor("extrudeType")
	fixedPath_ : FixedPathPlug = PlugDescriptor("fixedPath")
	length_ : LengthPlug = PlugDescriptor("length")
	outputSurface_ : OutputSurfacePlug = PlugDescriptor("outputSurface")
	path_ : PathPlug = PlugDescriptor("path")
	pivotX_ : PivotXPlug = PlugDescriptor("pivotX")
	pivotY_ : PivotYPlug = PlugDescriptor("pivotY")
	pivotZ_ : PivotZPlug = PlugDescriptor("pivotZ")
	pivot_ : PivotPlug = PlugDescriptor("pivot")
	profile_ : ProfilePlug = PlugDescriptor("profile")
	reverseSurfaceIfPathReversed_ : ReverseSurfaceIfPathReversedPlug = PlugDescriptor("reverseSurfaceIfPathReversed")
	rotation_ : RotationPlug = PlugDescriptor("rotation")
	scale_ : ScalePlug = PlugDescriptor("scale")
	subCurveSubSurface_ : SubCurveSubSurfacePlug = PlugDescriptor("subCurveSubSurface")
	useComponentPivot_ : UseComponentPivotPlug = PlugDescriptor("useComponentPivot")
	useProfileNormal_ : UseProfileNormalPlug = PlugDescriptor("useProfileNormal")

	# node attributes

	typeName = "extrude"
	apiTypeInt = 67
	apiTypeStr = "kExtrude"
	typeIdInt = 1313167442
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["degreeAlongLength", "directionX", "directionY", "directionZ", "direction", "extrudeType", "fixedPath", "length", "outputSurface", "path", "pivotX", "pivotY", "pivotZ", "pivot", "profile", "reverseSurfaceIfPathReversed", "rotation", "scale", "subCurveSubSurface", "useComponentPivot", "useProfileNormal"]
	nodeLeafPlugs = ["degreeAlongLength", "direction", "extrudeType", "fixedPath", "length", "outputSurface", "path", "pivot", "profile", "reverseSurfaceIfPathReversed", "rotation", "scale", "subCurveSubSurface", "useComponentPivot", "useProfileNormal"]
	pass

