

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifierUV = retriever.getNodeCls("PolyModifierUV")
assert PolyModifierUV
if T.TYPE_CHECKING:
	from .. import PolyModifierUV

# add node doc



# region plug type defs
class DenseLayoutPlug(Plug):
	node : PolyAutoProj = None
	pass
class LayoutPlug(Plug):
	node : PolyAutoProj = None
	pass
class LayoutMethodPlug(Plug):
	node : PolyAutoProj = None
	pass
class MaintainSymmetryPlug(Plug):
	node : PolyAutoProj = None
	pass
class OptimizePlug(Plug):
	node : PolyAutoProj = None
	pass
class PercentageSpacePlug(Plug):
	node : PolyAutoProj = None
	pass
class PivotXPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : PolyAutoProj = None
	pass
class PivotYPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : PolyAutoProj = None
	pass
class PivotZPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : PolyAutoProj = None
	pass
class PivotPlug(Plug):
	pivotX_ : PivotXPlug = PlugDescriptor("pivotX")
	pvx_ : PivotXPlug = PlugDescriptor("pivotX")
	pivotY_ : PivotYPlug = PlugDescriptor("pivotY")
	pvy_ : PivotYPlug = PlugDescriptor("pivotY")
	pivotZ_ : PivotZPlug = PlugDescriptor("pivotZ")
	pvz_ : PivotZPlug = PlugDescriptor("pivotZ")
	node : PolyAutoProj = None
	pass
class PlanesPlug(Plug):
	node : PolyAutoProj = None
	pass
class PolyGeomObjectPlug(Plug):
	node : PolyAutoProj = None
	pass
class ProjectBothDirectionsPlug(Plug):
	node : PolyAutoProj = None
	pass
class RotateXPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : PolyAutoProj = None
	pass
class RotateYPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : PolyAutoProj = None
	pass
class RotateZPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : PolyAutoProj = None
	pass
class RotatePlug(Plug):
	rotateX_ : RotateXPlug = PlugDescriptor("rotateX")
	rx_ : RotateXPlug = PlugDescriptor("rotateX")
	rotateY_ : RotateYPlug = PlugDescriptor("rotateY")
	ry_ : RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_ : RotateZPlug = PlugDescriptor("rotateZ")
	rz_ : RotateZPlug = PlugDescriptor("rotateZ")
	node : PolyAutoProj = None
	pass
class ScaleXPlug(Plug):
	parent : ScalePlug = PlugDescriptor("scale")
	node : PolyAutoProj = None
	pass
class ScaleYPlug(Plug):
	parent : ScalePlug = PlugDescriptor("scale")
	node : PolyAutoProj = None
	pass
class ScaleZPlug(Plug):
	parent : ScalePlug = PlugDescriptor("scale")
	node : PolyAutoProj = None
	pass
class ScalePlug(Plug):
	scaleX_ : ScaleXPlug = PlugDescriptor("scaleX")
	sx_ : ScaleXPlug = PlugDescriptor("scaleX")
	scaleY_ : ScaleYPlug = PlugDescriptor("scaleY")
	sy_ : ScaleYPlug = PlugDescriptor("scaleY")
	scaleZ_ : ScaleZPlug = PlugDescriptor("scaleZ")
	sz_ : ScaleZPlug = PlugDescriptor("scaleZ")
	node : PolyAutoProj = None
	pass
class ScaleModePlug(Plug):
	node : PolyAutoProj = None
	pass
class SkipIntersectPlug(Plug):
	node : PolyAutoProj = None
	pass
class TranslateXPlug(Plug):
	parent : TranslatePlug = PlugDescriptor("translate")
	node : PolyAutoProj = None
	pass
class TranslateYPlug(Plug):
	parent : TranslatePlug = PlugDescriptor("translate")
	node : PolyAutoProj = None
	pass
class TranslateZPlug(Plug):
	parent : TranslatePlug = PlugDescriptor("translate")
	node : PolyAutoProj = None
	pass
class TranslatePlug(Plug):
	translateX_ : TranslateXPlug = PlugDescriptor("translateX")
	tx_ : TranslateXPlug = PlugDescriptor("translateX")
	translateY_ : TranslateYPlug = PlugDescriptor("translateY")
	ty_ : TranslateYPlug = PlugDescriptor("translateY")
	translateZ_ : TranslateZPlug = PlugDescriptor("translateZ")
	tz_ : TranslateZPlug = PlugDescriptor("translateZ")
	node : PolyAutoProj = None
	pass
class TwoSidedLayoutPlug(Plug):
	node : PolyAutoProj = None
	pass
# endregion


# define node class
class PolyAutoProj(PolyModifierUV):
	denseLayout_ : DenseLayoutPlug = PlugDescriptor("denseLayout")
	layout_ : LayoutPlug = PlugDescriptor("layout")
	layoutMethod_ : LayoutMethodPlug = PlugDescriptor("layoutMethod")
	maintainSymmetry_ : MaintainSymmetryPlug = PlugDescriptor("maintainSymmetry")
	optimize_ : OptimizePlug = PlugDescriptor("optimize")
	percentageSpace_ : PercentageSpacePlug = PlugDescriptor("percentageSpace")
	pivotX_ : PivotXPlug = PlugDescriptor("pivotX")
	pivotY_ : PivotYPlug = PlugDescriptor("pivotY")
	pivotZ_ : PivotZPlug = PlugDescriptor("pivotZ")
	pivot_ : PivotPlug = PlugDescriptor("pivot")
	planes_ : PlanesPlug = PlugDescriptor("planes")
	polyGeomObject_ : PolyGeomObjectPlug = PlugDescriptor("polyGeomObject")
	projectBothDirections_ : ProjectBothDirectionsPlug = PlugDescriptor("projectBothDirections")
	rotateX_ : RotateXPlug = PlugDescriptor("rotateX")
	rotateY_ : RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_ : RotateZPlug = PlugDescriptor("rotateZ")
	rotate_ : RotatePlug = PlugDescriptor("rotate")
	scaleX_ : ScaleXPlug = PlugDescriptor("scaleX")
	scaleY_ : ScaleYPlug = PlugDescriptor("scaleY")
	scaleZ_ : ScaleZPlug = PlugDescriptor("scaleZ")
	scale_ : ScalePlug = PlugDescriptor("scale")
	scaleMode_ : ScaleModePlug = PlugDescriptor("scaleMode")
	skipIntersect_ : SkipIntersectPlug = PlugDescriptor("skipIntersect")
	translateX_ : TranslateXPlug = PlugDescriptor("translateX")
	translateY_ : TranslateYPlug = PlugDescriptor("translateY")
	translateZ_ : TranslateZPlug = PlugDescriptor("translateZ")
	translate_ : TranslatePlug = PlugDescriptor("translate")
	twoSidedLayout_ : TwoSidedLayoutPlug = PlugDescriptor("twoSidedLayout")

	# node attributes

	typeName = "polyAutoProj"
	apiTypeInt = 851
	apiTypeStr = "kPolyAutoProj"
	typeIdInt = 1346458960
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["denseLayout", "layout", "layoutMethod", "maintainSymmetry", "optimize", "percentageSpace", "pivotX", "pivotY", "pivotZ", "pivot", "planes", "polyGeomObject", "projectBothDirections", "rotateX", "rotateY", "rotateZ", "rotate", "scaleX", "scaleY", "scaleZ", "scale", "scaleMode", "skipIntersect", "translateX", "translateY", "translateZ", "translate", "twoSidedLayout"]
	nodeLeafPlugs = ["denseLayout", "layout", "layoutMethod", "maintainSymmetry", "optimize", "percentageSpace", "pivot", "planes", "polyGeomObject", "projectBothDirections", "rotate", "scale", "scaleMode", "skipIntersect", "translate", "twoSidedLayout"]
	pass

