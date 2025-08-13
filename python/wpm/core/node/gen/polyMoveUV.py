

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifier = retriever.getNodeCls("PolyModifier")
assert PolyModifier
if T.TYPE_CHECKING:
	from .. import PolyModifier

# add node doc



# region plug type defs
class AxisLenXPlug(Plug):
	parent : AxisLenPlug = PlugDescriptor("axisLen")
	node : PolyMoveUV = None
	pass
class AxisLenYPlug(Plug):
	parent : AxisLenPlug = PlugDescriptor("axisLen")
	node : PolyMoveUV = None
	pass
class AxisLenPlug(Plug):
	axisLenX_ : AxisLenXPlug = PlugDescriptor("axisLenX")
	lx_ : AxisLenXPlug = PlugDescriptor("axisLenX")
	axisLenY_ : AxisLenYPlug = PlugDescriptor("axisLenY")
	ly_ : AxisLenYPlug = PlugDescriptor("axisLenY")
	node : PolyMoveUV = None
	pass
class CompIdPlug(Plug):
	node : PolyMoveUV = None
	pass
class PivotUPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : PolyMoveUV = None
	pass
class PivotVPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : PolyMoveUV = None
	pass
class PivotPlug(Plug):
	pivotU_ : PivotUPlug = PlugDescriptor("pivotU")
	pvu_ : PivotUPlug = PlugDescriptor("pivotU")
	pivotV_ : PivotVPlug = PlugDescriptor("pivotV")
	pvv_ : PivotVPlug = PlugDescriptor("pivotV")
	node : PolyMoveUV = None
	pass
class RandomPlug(Plug):
	node : PolyMoveUV = None
	pass
class RandomSeedPlug(Plug):
	node : PolyMoveUV = None
	pass
class RotationAnglePlug(Plug):
	node : PolyMoveUV = None
	pass
class ScaleUPlug(Plug):
	parent : ScalePlug = PlugDescriptor("scale")
	node : PolyMoveUV = None
	pass
class ScaleVPlug(Plug):
	parent : ScalePlug = PlugDescriptor("scale")
	node : PolyMoveUV = None
	pass
class ScalePlug(Plug):
	scaleU_ : ScaleUPlug = PlugDescriptor("scaleU")
	su_ : ScaleUPlug = PlugDescriptor("scaleU")
	scaleV_ : ScaleVPlug = PlugDescriptor("scaleV")
	sv_ : ScaleVPlug = PlugDescriptor("scaleV")
	node : PolyMoveUV = None
	pass
class TranslateUPlug(Plug):
	parent : TranslatePlug = PlugDescriptor("translate")
	node : PolyMoveUV = None
	pass
class TranslateVPlug(Plug):
	parent : TranslatePlug = PlugDescriptor("translate")
	node : PolyMoveUV = None
	pass
class TranslatePlug(Plug):
	translateU_ : TranslateUPlug = PlugDescriptor("translateU")
	tu_ : TranslateUPlug = PlugDescriptor("translateU")
	translateV_ : TranslateVPlug = PlugDescriptor("translateV")
	tv_ : TranslateVPlug = PlugDescriptor("translateV")
	node : PolyMoveUV = None
	pass
class UvSetNamePlug(Plug):
	node : PolyMoveUV = None
	pass
# endregion


# define node class
class PolyMoveUV(PolyModifier):
	axisLenX_ : AxisLenXPlug = PlugDescriptor("axisLenX")
	axisLenY_ : AxisLenYPlug = PlugDescriptor("axisLenY")
	axisLen_ : AxisLenPlug = PlugDescriptor("axisLen")
	compId_ : CompIdPlug = PlugDescriptor("compId")
	pivotU_ : PivotUPlug = PlugDescriptor("pivotU")
	pivotV_ : PivotVPlug = PlugDescriptor("pivotV")
	pivot_ : PivotPlug = PlugDescriptor("pivot")
	random_ : RandomPlug = PlugDescriptor("random")
	randomSeed_ : RandomSeedPlug = PlugDescriptor("randomSeed")
	rotationAngle_ : RotationAnglePlug = PlugDescriptor("rotationAngle")
	scaleU_ : ScaleUPlug = PlugDescriptor("scaleU")
	scaleV_ : ScaleVPlug = PlugDescriptor("scaleV")
	scale_ : ScalePlug = PlugDescriptor("scale")
	translateU_ : TranslateUPlug = PlugDescriptor("translateU")
	translateV_ : TranslateVPlug = PlugDescriptor("translateV")
	translate_ : TranslatePlug = PlugDescriptor("translate")
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")

	# node attributes

	typeName = "polyMoveUV"
	apiTypeInt = 421
	apiTypeStr = "kPolyMoveUV"
	typeIdInt = 1347245398
	MFnCls = om.MFnDependencyNode
	pass

