

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifierWorld = Catalogue.PolyModifierWorld
else:
	from .. import retriever
	PolyModifierWorld = retriever.getNodeCls("PolyModifierWorld")
	assert PolyModifierWorld

# add node doc



# region plug type defs
class CompIdPlug(Plug):
	node : PolyMoveVertex = None
	pass
class GainPlug(Plug):
	node : PolyMoveVertex = None
	pass
class LocalDirectionXPlug(Plug):
	parent : LocalDirectionPlug = PlugDescriptor("localDirection")
	node : PolyMoveVertex = None
	pass
class LocalDirectionYPlug(Plug):
	parent : LocalDirectionPlug = PlugDescriptor("localDirection")
	node : PolyMoveVertex = None
	pass
class LocalDirectionZPlug(Plug):
	parent : LocalDirectionPlug = PlugDescriptor("localDirection")
	node : PolyMoveVertex = None
	pass
class LocalDirectionPlug(Plug):
	localDirectionX_ : LocalDirectionXPlug = PlugDescriptor("localDirectionX")
	ldx_ : LocalDirectionXPlug = PlugDescriptor("localDirectionX")
	localDirectionY_ : LocalDirectionYPlug = PlugDescriptor("localDirectionY")
	ldy_ : LocalDirectionYPlug = PlugDescriptor("localDirectionY")
	localDirectionZ_ : LocalDirectionZPlug = PlugDescriptor("localDirectionZ")
	ldz_ : LocalDirectionZPlug = PlugDescriptor("localDirectionZ")
	node : PolyMoveVertex = None
	pass
class LocalTranslateXPlug(Plug):
	parent : LocalTranslatePlug = PlugDescriptor("localTranslate")
	node : PolyMoveVertex = None
	pass
class LocalTranslateYPlug(Plug):
	parent : LocalTranslatePlug = PlugDescriptor("localTranslate")
	node : PolyMoveVertex = None
	pass
class LocalTranslateZPlug(Plug):
	parent : LocalTranslatePlug = PlugDescriptor("localTranslate")
	node : PolyMoveVertex = None
	pass
class LocalTranslatePlug(Plug):
	localTranslateX_ : LocalTranslateXPlug = PlugDescriptor("localTranslateX")
	ltx_ : LocalTranslateXPlug = PlugDescriptor("localTranslateX")
	localTranslateY_ : LocalTranslateYPlug = PlugDescriptor("localTranslateY")
	lty_ : LocalTranslateYPlug = PlugDescriptor("localTranslateY")
	localTranslateZ_ : LocalTranslateZPlug = PlugDescriptor("localTranslateZ")
	ltz_ : LocalTranslateZPlug = PlugDescriptor("localTranslateZ")
	node : PolyMoveVertex = None
	pass
class MatrixPlug(Plug):
	node : PolyMoveVertex = None
	pass
class PivotXPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : PolyMoveVertex = None
	pass
class PivotYPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : PolyMoveVertex = None
	pass
class PivotZPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : PolyMoveVertex = None
	pass
class PivotPlug(Plug):
	pivotX_ : PivotXPlug = PlugDescriptor("pivotX")
	pvx_ : PivotXPlug = PlugDescriptor("pivotX")
	pivotY_ : PivotYPlug = PlugDescriptor("pivotY")
	pvy_ : PivotYPlug = PlugDescriptor("pivotY")
	pivotZ_ : PivotZPlug = PlugDescriptor("pivotZ")
	pvz_ : PivotZPlug = PlugDescriptor("pivotZ")
	node : PolyMoveVertex = None
	pass
class RandomPlug(Plug):
	node : PolyMoveVertex = None
	pass
class RandomSeedPlug(Plug):
	node : PolyMoveVertex = None
	pass
class RotateXPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : PolyMoveVertex = None
	pass
class RotateYPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : PolyMoveVertex = None
	pass
class RotateZPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : PolyMoveVertex = None
	pass
class RotatePlug(Plug):
	rotateX_ : RotateXPlug = PlugDescriptor("rotateX")
	rx_ : RotateXPlug = PlugDescriptor("rotateX")
	rotateY_ : RotateYPlug = PlugDescriptor("rotateY")
	ry_ : RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_ : RotateZPlug = PlugDescriptor("rotateZ")
	rz_ : RotateZPlug = PlugDescriptor("rotateZ")
	node : PolyMoveVertex = None
	pass
class ScaleXPlug(Plug):
	parent : ScalePlug = PlugDescriptor("scale")
	node : PolyMoveVertex = None
	pass
class ScaleYPlug(Plug):
	parent : ScalePlug = PlugDescriptor("scale")
	node : PolyMoveVertex = None
	pass
class ScaleZPlug(Plug):
	parent : ScalePlug = PlugDescriptor("scale")
	node : PolyMoveVertex = None
	pass
class ScalePlug(Plug):
	scaleX_ : ScaleXPlug = PlugDescriptor("scaleX")
	sx_ : ScaleXPlug = PlugDescriptor("scaleX")
	scaleY_ : ScaleYPlug = PlugDescriptor("scaleY")
	sy_ : ScaleYPlug = PlugDescriptor("scaleY")
	scaleZ_ : ScaleZPlug = PlugDescriptor("scaleZ")
	sz_ : ScaleZPlug = PlugDescriptor("scaleZ")
	node : PolyMoveVertex = None
	pass
class TranslateXPlug(Plug):
	parent : TranslatePlug = PlugDescriptor("translate")
	node : PolyMoveVertex = None
	pass
class TranslateYPlug(Plug):
	parent : TranslatePlug = PlugDescriptor("translate")
	node : PolyMoveVertex = None
	pass
class TranslateZPlug(Plug):
	parent : TranslatePlug = PlugDescriptor("translate")
	node : PolyMoveVertex = None
	pass
class TranslatePlug(Plug):
	translateX_ : TranslateXPlug = PlugDescriptor("translateX")
	tx_ : TranslateXPlug = PlugDescriptor("translateX")
	translateY_ : TranslateYPlug = PlugDescriptor("translateY")
	ty_ : TranslateYPlug = PlugDescriptor("translateY")
	translateZ_ : TranslateZPlug = PlugDescriptor("translateZ")
	tz_ : TranslateZPlug = PlugDescriptor("translateZ")
	node : PolyMoveVertex = None
	pass
# endregion


# define node class
class PolyMoveVertex(PolyModifierWorld):
	compId_ : CompIdPlug = PlugDescriptor("compId")
	gain_ : GainPlug = PlugDescriptor("gain")
	localDirectionX_ : LocalDirectionXPlug = PlugDescriptor("localDirectionX")
	localDirectionY_ : LocalDirectionYPlug = PlugDescriptor("localDirectionY")
	localDirectionZ_ : LocalDirectionZPlug = PlugDescriptor("localDirectionZ")
	localDirection_ : LocalDirectionPlug = PlugDescriptor("localDirection")
	localTranslateX_ : LocalTranslateXPlug = PlugDescriptor("localTranslateX")
	localTranslateY_ : LocalTranslateYPlug = PlugDescriptor("localTranslateY")
	localTranslateZ_ : LocalTranslateZPlug = PlugDescriptor("localTranslateZ")
	localTranslate_ : LocalTranslatePlug = PlugDescriptor("localTranslate")
	matrix_ : MatrixPlug = PlugDescriptor("matrix")
	pivotX_ : PivotXPlug = PlugDescriptor("pivotX")
	pivotY_ : PivotYPlug = PlugDescriptor("pivotY")
	pivotZ_ : PivotZPlug = PlugDescriptor("pivotZ")
	pivot_ : PivotPlug = PlugDescriptor("pivot")
	random_ : RandomPlug = PlugDescriptor("random")
	randomSeed_ : RandomSeedPlug = PlugDescriptor("randomSeed")
	rotateX_ : RotateXPlug = PlugDescriptor("rotateX")
	rotateY_ : RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_ : RotateZPlug = PlugDescriptor("rotateZ")
	rotate_ : RotatePlug = PlugDescriptor("rotate")
	scaleX_ : ScaleXPlug = PlugDescriptor("scaleX")
	scaleY_ : ScaleYPlug = PlugDescriptor("scaleY")
	scaleZ_ : ScaleZPlug = PlugDescriptor("scaleZ")
	scale_ : ScalePlug = PlugDescriptor("scale")
	translateX_ : TranslateXPlug = PlugDescriptor("translateX")
	translateY_ : TranslateYPlug = PlugDescriptor("translateY")
	translateZ_ : TranslateZPlug = PlugDescriptor("translateZ")
	translate_ : TranslatePlug = PlugDescriptor("translate")

	# node attributes

	typeName = "polyMoveVertex"
	apiTypeInt = 422
	apiTypeStr = "kPolyMoveVertex"
	typeIdInt = 1347243862
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["compId", "gain", "localDirectionX", "localDirectionY", "localDirectionZ", "localDirection", "localTranslateX", "localTranslateY", "localTranslateZ", "localTranslate", "matrix", "pivotX", "pivotY", "pivotZ", "pivot", "random", "randomSeed", "rotateX", "rotateY", "rotateZ", "rotate", "scaleX", "scaleY", "scaleZ", "scale", "translateX", "translateY", "translateZ", "translate"]
	nodeLeafPlugs = ["compId", "gain", "localDirection", "localTranslate", "matrix", "pivot", "random", "randomSeed", "rotate", "scale", "translate"]
	pass

