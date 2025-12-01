

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
class ApplyPlug(Plug):
	node : AnimCurve = None
	pass
class BinMembershipPlug(Plug):
	node : AnimCurve = None
	pass
class CurveColorBPlug(Plug):
	parent : CurveColorPlug = PlugDescriptor("curveColor")
	node : AnimCurve = None
	pass
class CurveColorGPlug(Plug):
	parent : CurveColorPlug = PlugDescriptor("curveColor")
	node : AnimCurve = None
	pass
class CurveColorRPlug(Plug):
	parent : CurveColorPlug = PlugDescriptor("curveColor")
	node : AnimCurve = None
	pass
class CurveColorPlug(Plug):
	curveColorB_ : CurveColorBPlug = PlugDescriptor("curveColorB")
	ccb_ : CurveColorBPlug = PlugDescriptor("curveColorB")
	curveColorG_ : CurveColorGPlug = PlugDescriptor("curveColorG")
	ccg_ : CurveColorGPlug = PlugDescriptor("curveColorG")
	curveColorR_ : CurveColorRPlug = PlugDescriptor("curveColorR")
	ccr_ : CurveColorRPlug = PlugDescriptor("curveColorR")
	node : AnimCurve = None
	pass
class InStippleRangePlug(Plug):
	node : AnimCurve = None
	pass
class KeyBreakdownPlug(Plug):
	node : AnimCurve = None
	pass
class KeyTanInTypePlug(Plug):
	node : AnimCurve = None
	pass
class KeyTanInXPlug(Plug):
	node : AnimCurve = None
	pass
class KeyTanInYPlug(Plug):
	node : AnimCurve = None
	pass
class KeyTanLockedPlug(Plug):
	node : AnimCurve = None
	pass
class KeyTanOutTypePlug(Plug):
	node : AnimCurve = None
	pass
class KeyTanOutXPlug(Plug):
	node : AnimCurve = None
	pass
class KeyTanOutYPlug(Plug):
	node : AnimCurve = None
	pass
class KeyTickDrawSpecialPlug(Plug):
	node : AnimCurve = None
	pass
class KeyWeightLockedPlug(Plug):
	node : AnimCurve = None
	pass
class OutStippleRangePlug(Plug):
	node : AnimCurve = None
	pass
class OutStippleThresholdPlug(Plug):
	node : AnimCurve = None
	pass
class PostInfinityPlug(Plug):
	node : AnimCurve = None
	pass
class PreInfinityPlug(Plug):
	node : AnimCurve = None
	pass
class RotationInterpolationPlug(Plug):
	node : AnimCurve = None
	pass
class StipplePatternPlug(Plug):
	node : AnimCurve = None
	pass
class StippleReversePlug(Plug):
	node : AnimCurve = None
	pass
class TangentTypePlug(Plug):
	node : AnimCurve = None
	pass
class UseCurveColorPlug(Plug):
	node : AnimCurve = None
	pass
class WeightedTangentsPlug(Plug):
	node : AnimCurve = None
	pass
# endregion


# define node class
class AnimCurve(_BASE_):
	apply_ : ApplyPlug = PlugDescriptor("apply")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	curveColorB_ : CurveColorBPlug = PlugDescriptor("curveColorB")
	curveColorG_ : CurveColorGPlug = PlugDescriptor("curveColorG")
	curveColorR_ : CurveColorRPlug = PlugDescriptor("curveColorR")
	curveColor_ : CurveColorPlug = PlugDescriptor("curveColor")
	inStippleRange_ : InStippleRangePlug = PlugDescriptor("inStippleRange")
	keyBreakdown_ : KeyBreakdownPlug = PlugDescriptor("keyBreakdown")
	keyTanInType_ : KeyTanInTypePlug = PlugDescriptor("keyTanInType")
	keyTanInX_ : KeyTanInXPlug = PlugDescriptor("keyTanInX")
	keyTanInY_ : KeyTanInYPlug = PlugDescriptor("keyTanInY")
	keyTanLocked_ : KeyTanLockedPlug = PlugDescriptor("keyTanLocked")
	keyTanOutType_ : KeyTanOutTypePlug = PlugDescriptor("keyTanOutType")
	keyTanOutX_ : KeyTanOutXPlug = PlugDescriptor("keyTanOutX")
	keyTanOutY_ : KeyTanOutYPlug = PlugDescriptor("keyTanOutY")
	keyTickDrawSpecial_ : KeyTickDrawSpecialPlug = PlugDescriptor("keyTickDrawSpecial")
	keyWeightLocked_ : KeyWeightLockedPlug = PlugDescriptor("keyWeightLocked")
	outStippleRange_ : OutStippleRangePlug = PlugDescriptor("outStippleRange")
	outStippleThreshold_ : OutStippleThresholdPlug = PlugDescriptor("outStippleThreshold")
	postInfinity_ : PostInfinityPlug = PlugDescriptor("postInfinity")
	preInfinity_ : PreInfinityPlug = PlugDescriptor("preInfinity")
	rotationInterpolation_ : RotationInterpolationPlug = PlugDescriptor("rotationInterpolation")
	stipplePattern_ : StipplePatternPlug = PlugDescriptor("stipplePattern")
	stippleReverse_ : StippleReversePlug = PlugDescriptor("stippleReverse")
	tangentType_ : TangentTypePlug = PlugDescriptor("tangentType")
	useCurveColor_ : UseCurveColorPlug = PlugDescriptor("useCurveColor")
	weightedTangents_ : WeightedTangentsPlug = PlugDescriptor("weightedTangents")

	# node attributes

	typeName = "animCurve"
	apiTypeInt = 7
	apiTypeStr = "kAnimCurve"
	typeIdInt = 1347240790
	MFnCls = om.MFnAnimCurve
	nodeLeafClassAttrs = ["apply", "binMembership", "curveColorB", "curveColorG", "curveColorR", "curveColor", "inStippleRange", "keyBreakdown", "keyTanInType", "keyTanInX", "keyTanInY", "keyTanLocked", "keyTanOutType", "keyTanOutX", "keyTanOutY", "keyTickDrawSpecial", "keyWeightLocked", "outStippleRange", "outStippleThreshold", "postInfinity", "preInfinity", "rotationInterpolation", "stipplePattern", "stippleReverse", "tangentType", "useCurveColor", "weightedTangents"]
	nodeLeafPlugs = ["apply", "binMembership", "curveColor", "inStippleRange", "keyBreakdown", "keyTanInType", "keyTanInX", "keyTanInY", "keyTanLocked", "keyTanOutType", "keyTanOutX", "keyTanOutY", "keyTickDrawSpecial", "keyWeightLocked", "outStippleRange", "outStippleThreshold", "postInfinity", "preInfinity", "rotationInterpolation", "stipplePattern", "stippleReverse", "tangentType", "useCurveColor", "weightedTangents"]
	pass

