

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
_BASE_ = retriever.getNodeCls("_BASE_")
assert _BASE_
if T.TYPE_CHECKING:
	from .. import _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : PairBlend = None
	pass
class CurrentDriverPlug(Plug):
	node : PairBlend = None
	pass
class InRotateX1Plug(Plug):
	parent : InRotate1Plug = PlugDescriptor("inRotate1")
	node : PairBlend = None
	pass
class InRotateY1Plug(Plug):
	parent : InRotate1Plug = PlugDescriptor("inRotate1")
	node : PairBlend = None
	pass
class InRotateZ1Plug(Plug):
	parent : InRotate1Plug = PlugDescriptor("inRotate1")
	node : PairBlend = None
	pass
class InRotate1Plug(Plug):
	inRotateX1_ : InRotateX1Plug = PlugDescriptor("inRotateX1")
	irx1_ : InRotateX1Plug = PlugDescriptor("inRotateX1")
	inRotateY1_ : InRotateY1Plug = PlugDescriptor("inRotateY1")
	iry1_ : InRotateY1Plug = PlugDescriptor("inRotateY1")
	inRotateZ1_ : InRotateZ1Plug = PlugDescriptor("inRotateZ1")
	irz1_ : InRotateZ1Plug = PlugDescriptor("inRotateZ1")
	node : PairBlend = None
	pass
class InRotateX2Plug(Plug):
	parent : InRotate2Plug = PlugDescriptor("inRotate2")
	node : PairBlend = None
	pass
class InRotateY2Plug(Plug):
	parent : InRotate2Plug = PlugDescriptor("inRotate2")
	node : PairBlend = None
	pass
class InRotateZ2Plug(Plug):
	parent : InRotate2Plug = PlugDescriptor("inRotate2")
	node : PairBlend = None
	pass
class InRotate2Plug(Plug):
	inRotateX2_ : InRotateX2Plug = PlugDescriptor("inRotateX2")
	irx2_ : InRotateX2Plug = PlugDescriptor("inRotateX2")
	inRotateY2_ : InRotateY2Plug = PlugDescriptor("inRotateY2")
	iry2_ : InRotateY2Plug = PlugDescriptor("inRotateY2")
	inRotateZ2_ : InRotateZ2Plug = PlugDescriptor("inRotateZ2")
	irz2_ : InRotateZ2Plug = PlugDescriptor("inRotateZ2")
	node : PairBlend = None
	pass
class InTranslateX1Plug(Plug):
	parent : InTranslate1Plug = PlugDescriptor("inTranslate1")
	node : PairBlend = None
	pass
class InTranslateY1Plug(Plug):
	parent : InTranslate1Plug = PlugDescriptor("inTranslate1")
	node : PairBlend = None
	pass
class InTranslateZ1Plug(Plug):
	parent : InTranslate1Plug = PlugDescriptor("inTranslate1")
	node : PairBlend = None
	pass
class InTranslate1Plug(Plug):
	inTranslateX1_ : InTranslateX1Plug = PlugDescriptor("inTranslateX1")
	itx1_ : InTranslateX1Plug = PlugDescriptor("inTranslateX1")
	inTranslateY1_ : InTranslateY1Plug = PlugDescriptor("inTranslateY1")
	ity1_ : InTranslateY1Plug = PlugDescriptor("inTranslateY1")
	inTranslateZ1_ : InTranslateZ1Plug = PlugDescriptor("inTranslateZ1")
	itz1_ : InTranslateZ1Plug = PlugDescriptor("inTranslateZ1")
	node : PairBlend = None
	pass
class InTranslateX2Plug(Plug):
	parent : InTranslate2Plug = PlugDescriptor("inTranslate2")
	node : PairBlend = None
	pass
class InTranslateY2Plug(Plug):
	parent : InTranslate2Plug = PlugDescriptor("inTranslate2")
	node : PairBlend = None
	pass
class InTranslateZ2Plug(Plug):
	parent : InTranslate2Plug = PlugDescriptor("inTranslate2")
	node : PairBlend = None
	pass
class InTranslate2Plug(Plug):
	inTranslateX2_ : InTranslateX2Plug = PlugDescriptor("inTranslateX2")
	itx2_ : InTranslateX2Plug = PlugDescriptor("inTranslateX2")
	inTranslateY2_ : InTranslateY2Plug = PlugDescriptor("inTranslateY2")
	ity2_ : InTranslateY2Plug = PlugDescriptor("inTranslateY2")
	inTranslateZ2_ : InTranslateZ2Plug = PlugDescriptor("inTranslateZ2")
	itz2_ : InTranslateZ2Plug = PlugDescriptor("inTranslateZ2")
	node : PairBlend = None
	pass
class OutRotateXPlug(Plug):
	parent : OutRotatePlug = PlugDescriptor("outRotate")
	node : PairBlend = None
	pass
class OutRotateYPlug(Plug):
	parent : OutRotatePlug = PlugDescriptor("outRotate")
	node : PairBlend = None
	pass
class OutRotateZPlug(Plug):
	parent : OutRotatePlug = PlugDescriptor("outRotate")
	node : PairBlend = None
	pass
class OutRotatePlug(Plug):
	outRotateX_ : OutRotateXPlug = PlugDescriptor("outRotateX")
	orx_ : OutRotateXPlug = PlugDescriptor("outRotateX")
	outRotateY_ : OutRotateYPlug = PlugDescriptor("outRotateY")
	ory_ : OutRotateYPlug = PlugDescriptor("outRotateY")
	outRotateZ_ : OutRotateZPlug = PlugDescriptor("outRotateZ")
	orz_ : OutRotateZPlug = PlugDescriptor("outRotateZ")
	node : PairBlend = None
	pass
class OutTranslateXPlug(Plug):
	parent : OutTranslatePlug = PlugDescriptor("outTranslate")
	node : PairBlend = None
	pass
class OutTranslateYPlug(Plug):
	parent : OutTranslatePlug = PlugDescriptor("outTranslate")
	node : PairBlend = None
	pass
class OutTranslateZPlug(Plug):
	parent : OutTranslatePlug = PlugDescriptor("outTranslate")
	node : PairBlend = None
	pass
class OutTranslatePlug(Plug):
	outTranslateX_ : OutTranslateXPlug = PlugDescriptor("outTranslateX")
	otx_ : OutTranslateXPlug = PlugDescriptor("outTranslateX")
	outTranslateY_ : OutTranslateYPlug = PlugDescriptor("outTranslateY")
	oty_ : OutTranslateYPlug = PlugDescriptor("outTranslateY")
	outTranslateZ_ : OutTranslateZPlug = PlugDescriptor("outTranslateZ")
	otz_ : OutTranslateZPlug = PlugDescriptor("outTranslateZ")
	node : PairBlend = None
	pass
class RotInterpolationPlug(Plug):
	node : PairBlend = None
	pass
class RotateModePlug(Plug):
	node : PairBlend = None
	pass
class RotateOrderPlug(Plug):
	node : PairBlend = None
	pass
class TranslateXModePlug(Plug):
	node : PairBlend = None
	pass
class TranslateYModePlug(Plug):
	node : PairBlend = None
	pass
class TranslateZModePlug(Plug):
	node : PairBlend = None
	pass
class WeightPlug(Plug):
	node : PairBlend = None
	pass
# endregion


# define node class
class PairBlend(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	currentDriver_ : CurrentDriverPlug = PlugDescriptor("currentDriver")
	inRotateX1_ : InRotateX1Plug = PlugDescriptor("inRotateX1")
	inRotateY1_ : InRotateY1Plug = PlugDescriptor("inRotateY1")
	inRotateZ1_ : InRotateZ1Plug = PlugDescriptor("inRotateZ1")
	inRotate1_ : InRotate1Plug = PlugDescriptor("inRotate1")
	inRotateX2_ : InRotateX2Plug = PlugDescriptor("inRotateX2")
	inRotateY2_ : InRotateY2Plug = PlugDescriptor("inRotateY2")
	inRotateZ2_ : InRotateZ2Plug = PlugDescriptor("inRotateZ2")
	inRotate2_ : InRotate2Plug = PlugDescriptor("inRotate2")
	inTranslateX1_ : InTranslateX1Plug = PlugDescriptor("inTranslateX1")
	inTranslateY1_ : InTranslateY1Plug = PlugDescriptor("inTranslateY1")
	inTranslateZ1_ : InTranslateZ1Plug = PlugDescriptor("inTranslateZ1")
	inTranslate1_ : InTranslate1Plug = PlugDescriptor("inTranslate1")
	inTranslateX2_ : InTranslateX2Plug = PlugDescriptor("inTranslateX2")
	inTranslateY2_ : InTranslateY2Plug = PlugDescriptor("inTranslateY2")
	inTranslateZ2_ : InTranslateZ2Plug = PlugDescriptor("inTranslateZ2")
	inTranslate2_ : InTranslate2Plug = PlugDescriptor("inTranslate2")
	outRotateX_ : OutRotateXPlug = PlugDescriptor("outRotateX")
	outRotateY_ : OutRotateYPlug = PlugDescriptor("outRotateY")
	outRotateZ_ : OutRotateZPlug = PlugDescriptor("outRotateZ")
	outRotate_ : OutRotatePlug = PlugDescriptor("outRotate")
	outTranslateX_ : OutTranslateXPlug = PlugDescriptor("outTranslateX")
	outTranslateY_ : OutTranslateYPlug = PlugDescriptor("outTranslateY")
	outTranslateZ_ : OutTranslateZPlug = PlugDescriptor("outTranslateZ")
	outTranslate_ : OutTranslatePlug = PlugDescriptor("outTranslate")
	rotInterpolation_ : RotInterpolationPlug = PlugDescriptor("rotInterpolation")
	rotateMode_ : RotateModePlug = PlugDescriptor("rotateMode")
	rotateOrder_ : RotateOrderPlug = PlugDescriptor("rotateOrder")
	translateXMode_ : TranslateXModePlug = PlugDescriptor("translateXMode")
	translateYMode_ : TranslateYModePlug = PlugDescriptor("translateYMode")
	translateZMode_ : TranslateZModePlug = PlugDescriptor("translateZMode")
	weight_ : WeightPlug = PlugDescriptor("weight")

	# node attributes

	typeName = "pairBlend"
	apiTypeInt = 927
	apiTypeStr = "kPairBlend"
	typeIdInt = 1095778892
	MFnCls = om.MFnDependencyNode
	pass

