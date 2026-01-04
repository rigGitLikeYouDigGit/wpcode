

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	DagNode = Catalogue.DagNode
else:
	from .. import retriever
	DagNode = retriever.getNodeCls("DagNode")
	assert DagNode

# add node doc



# region plug type defs
class DagLocalInverseMatrixPlug(Plug):
	node : Transform = None
	pass
class DagLocalMatrixPlug(Plug):
	node : Transform = None
	pass
class DisplayHandlePlug(Plug):
	node : Transform = None
	pass
class DisplayLocalAxisPlug(Plug):
	node : Transform = None
	pass
class DisplayRotatePivotPlug(Plug):
	node : Transform = None
	pass
class DisplayScalePivotPlug(Plug):
	node : Transform = None
	pass
class DynamicsPlug(Plug):
	node : Transform = None
	pass
class GeometryPlug(Plug):
	node : Transform = None
	pass
class InheritsTransformPlug(Plug):
	node : Transform = None
	pass
class MaxRotXLimitPlug(Plug):
	parent : MaxRotLimitPlug = PlugDescriptor("maxRotLimit")
	node : Transform = None
	pass
class MaxRotYLimitPlug(Plug):
	parent : MaxRotLimitPlug = PlugDescriptor("maxRotLimit")
	node : Transform = None
	pass
class MaxRotZLimitPlug(Plug):
	parent : MaxRotLimitPlug = PlugDescriptor("maxRotLimit")
	node : Transform = None
	pass
class MaxRotLimitPlug(Plug):
	maxRotXLimit_ : MaxRotXLimitPlug = PlugDescriptor("maxRotXLimit")
	xrxl_ : MaxRotXLimitPlug = PlugDescriptor("maxRotXLimit")
	maxRotYLimit_ : MaxRotYLimitPlug = PlugDescriptor("maxRotYLimit")
	xryl_ : MaxRotYLimitPlug = PlugDescriptor("maxRotYLimit")
	maxRotZLimit_ : MaxRotZLimitPlug = PlugDescriptor("maxRotZLimit")
	xrzl_ : MaxRotZLimitPlug = PlugDescriptor("maxRotZLimit")
	node : Transform = None
	pass
class MaxRotXLimitEnablePlug(Plug):
	parent : MaxRotLimitEnablePlug = PlugDescriptor("maxRotLimitEnable")
	node : Transform = None
	pass
class MaxRotYLimitEnablePlug(Plug):
	parent : MaxRotLimitEnablePlug = PlugDescriptor("maxRotLimitEnable")
	node : Transform = None
	pass
class MaxRotZLimitEnablePlug(Plug):
	parent : MaxRotLimitEnablePlug = PlugDescriptor("maxRotLimitEnable")
	node : Transform = None
	pass
class MaxRotLimitEnablePlug(Plug):
	maxRotXLimitEnable_ : MaxRotXLimitEnablePlug = PlugDescriptor("maxRotXLimitEnable")
	xrxe_ : MaxRotXLimitEnablePlug = PlugDescriptor("maxRotXLimitEnable")
	maxRotYLimitEnable_ : MaxRotYLimitEnablePlug = PlugDescriptor("maxRotYLimitEnable")
	xrye_ : MaxRotYLimitEnablePlug = PlugDescriptor("maxRotYLimitEnable")
	maxRotZLimitEnable_ : MaxRotZLimitEnablePlug = PlugDescriptor("maxRotZLimitEnable")
	xrze_ : MaxRotZLimitEnablePlug = PlugDescriptor("maxRotZLimitEnable")
	node : Transform = None
	pass
class MaxScaleXLimitPlug(Plug):
	parent : MaxScaleLimitPlug = PlugDescriptor("maxScaleLimit")
	node : Transform = None
	pass
class MaxScaleYLimitPlug(Plug):
	parent : MaxScaleLimitPlug = PlugDescriptor("maxScaleLimit")
	node : Transform = None
	pass
class MaxScaleZLimitPlug(Plug):
	parent : MaxScaleLimitPlug = PlugDescriptor("maxScaleLimit")
	node : Transform = None
	pass
class MaxScaleLimitPlug(Plug):
	maxScaleXLimit_ : MaxScaleXLimitPlug = PlugDescriptor("maxScaleXLimit")
	xsxl_ : MaxScaleXLimitPlug = PlugDescriptor("maxScaleXLimit")
	maxScaleYLimit_ : MaxScaleYLimitPlug = PlugDescriptor("maxScaleYLimit")
	xsyl_ : MaxScaleYLimitPlug = PlugDescriptor("maxScaleYLimit")
	maxScaleZLimit_ : MaxScaleZLimitPlug = PlugDescriptor("maxScaleZLimit")
	xszl_ : MaxScaleZLimitPlug = PlugDescriptor("maxScaleZLimit")
	node : Transform = None
	pass
class MaxScaleXLimitEnablePlug(Plug):
	parent : MaxScaleLimitEnablePlug = PlugDescriptor("maxScaleLimitEnable")
	node : Transform = None
	pass
class MaxScaleYLimitEnablePlug(Plug):
	parent : MaxScaleLimitEnablePlug = PlugDescriptor("maxScaleLimitEnable")
	node : Transform = None
	pass
class MaxScaleZLimitEnablePlug(Plug):
	parent : MaxScaleLimitEnablePlug = PlugDescriptor("maxScaleLimitEnable")
	node : Transform = None
	pass
class MaxScaleLimitEnablePlug(Plug):
	maxScaleXLimitEnable_ : MaxScaleXLimitEnablePlug = PlugDescriptor("maxScaleXLimitEnable")
	xsxe_ : MaxScaleXLimitEnablePlug = PlugDescriptor("maxScaleXLimitEnable")
	maxScaleYLimitEnable_ : MaxScaleYLimitEnablePlug = PlugDescriptor("maxScaleYLimitEnable")
	xsye_ : MaxScaleYLimitEnablePlug = PlugDescriptor("maxScaleYLimitEnable")
	maxScaleZLimitEnable_ : MaxScaleZLimitEnablePlug = PlugDescriptor("maxScaleZLimitEnable")
	xsze_ : MaxScaleZLimitEnablePlug = PlugDescriptor("maxScaleZLimitEnable")
	node : Transform = None
	pass
class MaxTransXLimitPlug(Plug):
	parent : MaxTransLimitPlug = PlugDescriptor("maxTransLimit")
	node : Transform = None
	pass
class MaxTransYLimitPlug(Plug):
	parent : MaxTransLimitPlug = PlugDescriptor("maxTransLimit")
	node : Transform = None
	pass
class MaxTransZLimitPlug(Plug):
	parent : MaxTransLimitPlug = PlugDescriptor("maxTransLimit")
	node : Transform = None
	pass
class MaxTransLimitPlug(Plug):
	maxTransXLimit_ : MaxTransXLimitPlug = PlugDescriptor("maxTransXLimit")
	xtxl_ : MaxTransXLimitPlug = PlugDescriptor("maxTransXLimit")
	maxTransYLimit_ : MaxTransYLimitPlug = PlugDescriptor("maxTransYLimit")
	xtyl_ : MaxTransYLimitPlug = PlugDescriptor("maxTransYLimit")
	maxTransZLimit_ : MaxTransZLimitPlug = PlugDescriptor("maxTransZLimit")
	xtzl_ : MaxTransZLimitPlug = PlugDescriptor("maxTransZLimit")
	node : Transform = None
	pass
class MaxTransXLimitEnablePlug(Plug):
	parent : MaxTransLimitEnablePlug = PlugDescriptor("maxTransLimitEnable")
	node : Transform = None
	pass
class MaxTransYLimitEnablePlug(Plug):
	parent : MaxTransLimitEnablePlug = PlugDescriptor("maxTransLimitEnable")
	node : Transform = None
	pass
class MaxTransZLimitEnablePlug(Plug):
	parent : MaxTransLimitEnablePlug = PlugDescriptor("maxTransLimitEnable")
	node : Transform = None
	pass
class MaxTransLimitEnablePlug(Plug):
	maxTransXLimitEnable_ : MaxTransXLimitEnablePlug = PlugDescriptor("maxTransXLimitEnable")
	xtxe_ : MaxTransXLimitEnablePlug = PlugDescriptor("maxTransXLimitEnable")
	maxTransYLimitEnable_ : MaxTransYLimitEnablePlug = PlugDescriptor("maxTransYLimitEnable")
	xtye_ : MaxTransYLimitEnablePlug = PlugDescriptor("maxTransYLimitEnable")
	maxTransZLimitEnable_ : MaxTransZLimitEnablePlug = PlugDescriptor("maxTransZLimitEnable")
	xtze_ : MaxTransZLimitEnablePlug = PlugDescriptor("maxTransZLimitEnable")
	node : Transform = None
	pass
class MinRotXLimitPlug(Plug):
	parent : MinRotLimitPlug = PlugDescriptor("minRotLimit")
	node : Transform = None
	pass
class MinRotYLimitPlug(Plug):
	parent : MinRotLimitPlug = PlugDescriptor("minRotLimit")
	node : Transform = None
	pass
class MinRotZLimitPlug(Plug):
	parent : MinRotLimitPlug = PlugDescriptor("minRotLimit")
	node : Transform = None
	pass
class MinRotLimitPlug(Plug):
	minRotXLimit_ : MinRotXLimitPlug = PlugDescriptor("minRotXLimit")
	mrxl_ : MinRotXLimitPlug = PlugDescriptor("minRotXLimit")
	minRotYLimit_ : MinRotYLimitPlug = PlugDescriptor("minRotYLimit")
	mryl_ : MinRotYLimitPlug = PlugDescriptor("minRotYLimit")
	minRotZLimit_ : MinRotZLimitPlug = PlugDescriptor("minRotZLimit")
	mrzl_ : MinRotZLimitPlug = PlugDescriptor("minRotZLimit")
	node : Transform = None
	pass
class MinRotXLimitEnablePlug(Plug):
	parent : MinRotLimitEnablePlug = PlugDescriptor("minRotLimitEnable")
	node : Transform = None
	pass
class MinRotYLimitEnablePlug(Plug):
	parent : MinRotLimitEnablePlug = PlugDescriptor("minRotLimitEnable")
	node : Transform = None
	pass
class MinRotZLimitEnablePlug(Plug):
	parent : MinRotLimitEnablePlug = PlugDescriptor("minRotLimitEnable")
	node : Transform = None
	pass
class MinRotLimitEnablePlug(Plug):
	minRotXLimitEnable_ : MinRotXLimitEnablePlug = PlugDescriptor("minRotXLimitEnable")
	mrxe_ : MinRotXLimitEnablePlug = PlugDescriptor("minRotXLimitEnable")
	minRotYLimitEnable_ : MinRotYLimitEnablePlug = PlugDescriptor("minRotYLimitEnable")
	mrye_ : MinRotYLimitEnablePlug = PlugDescriptor("minRotYLimitEnable")
	minRotZLimitEnable_ : MinRotZLimitEnablePlug = PlugDescriptor("minRotZLimitEnable")
	mrze_ : MinRotZLimitEnablePlug = PlugDescriptor("minRotZLimitEnable")
	node : Transform = None
	pass
class MinScaleXLimitPlug(Plug):
	parent : MinScaleLimitPlug = PlugDescriptor("minScaleLimit")
	node : Transform = None
	pass
class MinScaleYLimitPlug(Plug):
	parent : MinScaleLimitPlug = PlugDescriptor("minScaleLimit")
	node : Transform = None
	pass
class MinScaleZLimitPlug(Plug):
	parent : MinScaleLimitPlug = PlugDescriptor("minScaleLimit")
	node : Transform = None
	pass
class MinScaleLimitPlug(Plug):
	minScaleXLimit_ : MinScaleXLimitPlug = PlugDescriptor("minScaleXLimit")
	msxl_ : MinScaleXLimitPlug = PlugDescriptor("minScaleXLimit")
	minScaleYLimit_ : MinScaleYLimitPlug = PlugDescriptor("minScaleYLimit")
	msyl_ : MinScaleYLimitPlug = PlugDescriptor("minScaleYLimit")
	minScaleZLimit_ : MinScaleZLimitPlug = PlugDescriptor("minScaleZLimit")
	mszl_ : MinScaleZLimitPlug = PlugDescriptor("minScaleZLimit")
	node : Transform = None
	pass
class MinScaleXLimitEnablePlug(Plug):
	parent : MinScaleLimitEnablePlug = PlugDescriptor("minScaleLimitEnable")
	node : Transform = None
	pass
class MinScaleYLimitEnablePlug(Plug):
	parent : MinScaleLimitEnablePlug = PlugDescriptor("minScaleLimitEnable")
	node : Transform = None
	pass
class MinScaleZLimitEnablePlug(Plug):
	parent : MinScaleLimitEnablePlug = PlugDescriptor("minScaleLimitEnable")
	node : Transform = None
	pass
class MinScaleLimitEnablePlug(Plug):
	minScaleXLimitEnable_ : MinScaleXLimitEnablePlug = PlugDescriptor("minScaleXLimitEnable")
	msxe_ : MinScaleXLimitEnablePlug = PlugDescriptor("minScaleXLimitEnable")
	minScaleYLimitEnable_ : MinScaleYLimitEnablePlug = PlugDescriptor("minScaleYLimitEnable")
	msye_ : MinScaleYLimitEnablePlug = PlugDescriptor("minScaleYLimitEnable")
	minScaleZLimitEnable_ : MinScaleZLimitEnablePlug = PlugDescriptor("minScaleZLimitEnable")
	msze_ : MinScaleZLimitEnablePlug = PlugDescriptor("minScaleZLimitEnable")
	node : Transform = None
	pass
class MinTransXLimitPlug(Plug):
	parent : MinTransLimitPlug = PlugDescriptor("minTransLimit")
	node : Transform = None
	pass
class MinTransYLimitPlug(Plug):
	parent : MinTransLimitPlug = PlugDescriptor("minTransLimit")
	node : Transform = None
	pass
class MinTransZLimitPlug(Plug):
	parent : MinTransLimitPlug = PlugDescriptor("minTransLimit")
	node : Transform = None
	pass
class MinTransLimitPlug(Plug):
	minTransXLimit_ : MinTransXLimitPlug = PlugDescriptor("minTransXLimit")
	mtxl_ : MinTransXLimitPlug = PlugDescriptor("minTransXLimit")
	minTransYLimit_ : MinTransYLimitPlug = PlugDescriptor("minTransYLimit")
	mtyl_ : MinTransYLimitPlug = PlugDescriptor("minTransYLimit")
	minTransZLimit_ : MinTransZLimitPlug = PlugDescriptor("minTransZLimit")
	mtzl_ : MinTransZLimitPlug = PlugDescriptor("minTransZLimit")
	node : Transform = None
	pass
class MinTransXLimitEnablePlug(Plug):
	parent : MinTransLimitEnablePlug = PlugDescriptor("minTransLimitEnable")
	node : Transform = None
	pass
class MinTransYLimitEnablePlug(Plug):
	parent : MinTransLimitEnablePlug = PlugDescriptor("minTransLimitEnable")
	node : Transform = None
	pass
class MinTransZLimitEnablePlug(Plug):
	parent : MinTransLimitEnablePlug = PlugDescriptor("minTransLimitEnable")
	node : Transform = None
	pass
class MinTransLimitEnablePlug(Plug):
	minTransXLimitEnable_ : MinTransXLimitEnablePlug = PlugDescriptor("minTransXLimitEnable")
	mtxe_ : MinTransXLimitEnablePlug = PlugDescriptor("minTransXLimitEnable")
	minTransYLimitEnable_ : MinTransYLimitEnablePlug = PlugDescriptor("minTransYLimitEnable")
	mtye_ : MinTransYLimitEnablePlug = PlugDescriptor("minTransYLimitEnable")
	minTransZLimitEnable_ : MinTransZLimitEnablePlug = PlugDescriptor("minTransZLimitEnable")
	mtze_ : MinTransZLimitEnablePlug = PlugDescriptor("minTransZLimitEnable")
	node : Transform = None
	pass
class OffsetParentMatrixPlug(Plug):
	node : Transform = None
	pass
class RotateXPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : Transform = None
	pass
class RotateYPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : Transform = None
	pass
class RotateZPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : Transform = None
	pass
class RotatePlug(Plug):
	rotateX_ : RotateXPlug = PlugDescriptor("rotateX")
	rx_ : RotateXPlug = PlugDescriptor("rotateX")
	rotateY_ : RotateYPlug = PlugDescriptor("rotateY")
	ry_ : RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_ : RotateZPlug = PlugDescriptor("rotateZ")
	rz_ : RotateZPlug = PlugDescriptor("rotateZ")
	node : Transform = None
	pass
class RotateAxisXPlug(Plug):
	parent : RotateAxisPlug = PlugDescriptor("rotateAxis")
	node : Transform = None
	pass
class RotateAxisYPlug(Plug):
	parent : RotateAxisPlug = PlugDescriptor("rotateAxis")
	node : Transform = None
	pass
class RotateAxisZPlug(Plug):
	parent : RotateAxisPlug = PlugDescriptor("rotateAxis")
	node : Transform = None
	pass
class RotateAxisPlug(Plug):
	rotateAxisX_ : RotateAxisXPlug = PlugDescriptor("rotateAxisX")
	rax_ : RotateAxisXPlug = PlugDescriptor("rotateAxisX")
	rotateAxisY_ : RotateAxisYPlug = PlugDescriptor("rotateAxisY")
	ray_ : RotateAxisYPlug = PlugDescriptor("rotateAxisY")
	rotateAxisZ_ : RotateAxisZPlug = PlugDescriptor("rotateAxisZ")
	raz_ : RotateAxisZPlug = PlugDescriptor("rotateAxisZ")
	node : Transform = None
	pass
class RotateOrderPlug(Plug):
	node : Transform = None
	pass
class RotatePivotXPlug(Plug):
	parent : RotatePivotPlug = PlugDescriptor("rotatePivot")
	node : Transform = None
	pass
class RotatePivotYPlug(Plug):
	parent : RotatePivotPlug = PlugDescriptor("rotatePivot")
	node : Transform = None
	pass
class RotatePivotZPlug(Plug):
	parent : RotatePivotPlug = PlugDescriptor("rotatePivot")
	node : Transform = None
	pass
class RotatePivotPlug(Plug):
	rotatePivotX_ : RotatePivotXPlug = PlugDescriptor("rotatePivotX")
	rpx_ : RotatePivotXPlug = PlugDescriptor("rotatePivotX")
	rotatePivotY_ : RotatePivotYPlug = PlugDescriptor("rotatePivotY")
	rpy_ : RotatePivotYPlug = PlugDescriptor("rotatePivotY")
	rotatePivotZ_ : RotatePivotZPlug = PlugDescriptor("rotatePivotZ")
	rpz_ : RotatePivotZPlug = PlugDescriptor("rotatePivotZ")
	node : Transform = None
	pass
class RotatePivotTranslateXPlug(Plug):
	parent : RotatePivotTranslatePlug = PlugDescriptor("rotatePivotTranslate")
	node : Transform = None
	pass
class RotatePivotTranslateYPlug(Plug):
	parent : RotatePivotTranslatePlug = PlugDescriptor("rotatePivotTranslate")
	node : Transform = None
	pass
class RotatePivotTranslateZPlug(Plug):
	parent : RotatePivotTranslatePlug = PlugDescriptor("rotatePivotTranslate")
	node : Transform = None
	pass
class RotatePivotTranslatePlug(Plug):
	rotatePivotTranslateX_ : RotatePivotTranslateXPlug = PlugDescriptor("rotatePivotTranslateX")
	rptx_ : RotatePivotTranslateXPlug = PlugDescriptor("rotatePivotTranslateX")
	rotatePivotTranslateY_ : RotatePivotTranslateYPlug = PlugDescriptor("rotatePivotTranslateY")
	rpty_ : RotatePivotTranslateYPlug = PlugDescriptor("rotatePivotTranslateY")
	rotatePivotTranslateZ_ : RotatePivotTranslateZPlug = PlugDescriptor("rotatePivotTranslateZ")
	rptz_ : RotatePivotTranslateZPlug = PlugDescriptor("rotatePivotTranslateZ")
	node : Transform = None
	pass
class RotateQuaternionWPlug(Plug):
	parent : RotateQuaternionPlug = PlugDescriptor("rotateQuaternion")
	node : Transform = None
	pass
class RotateQuaternionXPlug(Plug):
	parent : RotateQuaternionPlug = PlugDescriptor("rotateQuaternion")
	node : Transform = None
	pass
class RotateQuaternionYPlug(Plug):
	parent : RotateQuaternionPlug = PlugDescriptor("rotateQuaternion")
	node : Transform = None
	pass
class RotateQuaternionZPlug(Plug):
	parent : RotateQuaternionPlug = PlugDescriptor("rotateQuaternion")
	node : Transform = None
	pass
class RotateQuaternionPlug(Plug):
	rotateQuaternionW_ : RotateQuaternionWPlug = PlugDescriptor("rotateQuaternionW")
	rqw_ : RotateQuaternionWPlug = PlugDescriptor("rotateQuaternionW")
	rotateQuaternionX_ : RotateQuaternionXPlug = PlugDescriptor("rotateQuaternionX")
	rqx_ : RotateQuaternionXPlug = PlugDescriptor("rotateQuaternionX")
	rotateQuaternionY_ : RotateQuaternionYPlug = PlugDescriptor("rotateQuaternionY")
	rqy_ : RotateQuaternionYPlug = PlugDescriptor("rotateQuaternionY")
	rotateQuaternionZ_ : RotateQuaternionZPlug = PlugDescriptor("rotateQuaternionZ")
	rqz_ : RotateQuaternionZPlug = PlugDescriptor("rotateQuaternionZ")
	node : Transform = None
	pass
class RotationInterpolationPlug(Plug):
	node : Transform = None
	pass
class ScaleXPlug(Plug):
	parent : ScalePlug = PlugDescriptor("scale")
	node : Transform = None
	pass
class ScaleYPlug(Plug):
	parent : ScalePlug = PlugDescriptor("scale")
	node : Transform = None
	pass
class ScaleZPlug(Plug):
	parent : ScalePlug = PlugDescriptor("scale")
	node : Transform = None
	pass
class ScalePlug(Plug):
	scaleX_ : ScaleXPlug = PlugDescriptor("scaleX")
	sx_ : ScaleXPlug = PlugDescriptor("scaleX")
	scaleY_ : ScaleYPlug = PlugDescriptor("scaleY")
	sy_ : ScaleYPlug = PlugDescriptor("scaleY")
	scaleZ_ : ScaleZPlug = PlugDescriptor("scaleZ")
	sz_ : ScaleZPlug = PlugDescriptor("scaleZ")
	node : Transform = None
	pass
class ScalePivotXPlug(Plug):
	parent : ScalePivotPlug = PlugDescriptor("scalePivot")
	node : Transform = None
	pass
class ScalePivotYPlug(Plug):
	parent : ScalePivotPlug = PlugDescriptor("scalePivot")
	node : Transform = None
	pass
class ScalePivotZPlug(Plug):
	parent : ScalePivotPlug = PlugDescriptor("scalePivot")
	node : Transform = None
	pass
class ScalePivotPlug(Plug):
	scalePivotX_ : ScalePivotXPlug = PlugDescriptor("scalePivotX")
	spx_ : ScalePivotXPlug = PlugDescriptor("scalePivotX")
	scalePivotY_ : ScalePivotYPlug = PlugDescriptor("scalePivotY")
	spy_ : ScalePivotYPlug = PlugDescriptor("scalePivotY")
	scalePivotZ_ : ScalePivotZPlug = PlugDescriptor("scalePivotZ")
	spz_ : ScalePivotZPlug = PlugDescriptor("scalePivotZ")
	node : Transform = None
	pass
class ScalePivotTranslateXPlug(Plug):
	parent : ScalePivotTranslatePlug = PlugDescriptor("scalePivotTranslate")
	node : Transform = None
	pass
class ScalePivotTranslateYPlug(Plug):
	parent : ScalePivotTranslatePlug = PlugDescriptor("scalePivotTranslate")
	node : Transform = None
	pass
class ScalePivotTranslateZPlug(Plug):
	parent : ScalePivotTranslatePlug = PlugDescriptor("scalePivotTranslate")
	node : Transform = None
	pass
class ScalePivotTranslatePlug(Plug):
	scalePivotTranslateX_ : ScalePivotTranslateXPlug = PlugDescriptor("scalePivotTranslateX")
	sptx_ : ScalePivotTranslateXPlug = PlugDescriptor("scalePivotTranslateX")
	scalePivotTranslateY_ : ScalePivotTranslateYPlug = PlugDescriptor("scalePivotTranslateY")
	spty_ : ScalePivotTranslateYPlug = PlugDescriptor("scalePivotTranslateY")
	scalePivotTranslateZ_ : ScalePivotTranslateZPlug = PlugDescriptor("scalePivotTranslateZ")
	sptz_ : ScalePivotTranslateZPlug = PlugDescriptor("scalePivotTranslateZ")
	node : Transform = None
	pass
class SelectHandleXPlug(Plug):
	parent : SelectHandlePlug = PlugDescriptor("selectHandle")
	node : Transform = None
	pass
class SelectHandleYPlug(Plug):
	parent : SelectHandlePlug = PlugDescriptor("selectHandle")
	node : Transform = None
	pass
class SelectHandleZPlug(Plug):
	parent : SelectHandlePlug = PlugDescriptor("selectHandle")
	node : Transform = None
	pass
class SelectHandlePlug(Plug):
	selectHandleX_ : SelectHandleXPlug = PlugDescriptor("selectHandleX")
	hdlx_ : SelectHandleXPlug = PlugDescriptor("selectHandleX")
	selectHandleY_ : SelectHandleYPlug = PlugDescriptor("selectHandleY")
	hdly_ : SelectHandleYPlug = PlugDescriptor("selectHandleY")
	selectHandleZ_ : SelectHandleZPlug = PlugDescriptor("selectHandleZ")
	hdlz_ : SelectHandleZPlug = PlugDescriptor("selectHandleZ")
	node : Transform = None
	pass
class ShearXYPlug(Plug):
	parent : ShearPlug = PlugDescriptor("shear")
	node : Transform = None
	pass
class ShearXZPlug(Plug):
	parent : ShearPlug = PlugDescriptor("shear")
	node : Transform = None
	pass
class ShearYZPlug(Plug):
	parent : ShearPlug = PlugDescriptor("shear")
	node : Transform = None
	pass
class ShearPlug(Plug):
	shearXY_ : ShearXYPlug = PlugDescriptor("shearXY")
	shxy_ : ShearXYPlug = PlugDescriptor("shearXY")
	shearXZ_ : ShearXZPlug = PlugDescriptor("shearXZ")
	shxz_ : ShearXZPlug = PlugDescriptor("shearXZ")
	shearYZ_ : ShearYZPlug = PlugDescriptor("shearYZ")
	shyz_ : ShearYZPlug = PlugDescriptor("shearYZ")
	node : Transform = None
	pass
class ShowManipDefaultPlug(Plug):
	node : Transform = None
	pass
class SpecifiedManipLocationPlug(Plug):
	node : Transform = None
	pass
class TransMinusRotatePivotXPlug(Plug):
	parent : TransMinusRotatePivotPlug = PlugDescriptor("transMinusRotatePivot")
	node : Transform = None
	pass
class TransMinusRotatePivotYPlug(Plug):
	parent : TransMinusRotatePivotPlug = PlugDescriptor("transMinusRotatePivot")
	node : Transform = None
	pass
class TransMinusRotatePivotZPlug(Plug):
	parent : TransMinusRotatePivotPlug = PlugDescriptor("transMinusRotatePivot")
	node : Transform = None
	pass
class TransMinusRotatePivotPlug(Plug):
	transMinusRotatePivotX_ : TransMinusRotatePivotXPlug = PlugDescriptor("transMinusRotatePivotX")
	tmrx_ : TransMinusRotatePivotXPlug = PlugDescriptor("transMinusRotatePivotX")
	transMinusRotatePivotY_ : TransMinusRotatePivotYPlug = PlugDescriptor("transMinusRotatePivotY")
	tmry_ : TransMinusRotatePivotYPlug = PlugDescriptor("transMinusRotatePivotY")
	transMinusRotatePivotZ_ : TransMinusRotatePivotZPlug = PlugDescriptor("transMinusRotatePivotZ")
	tmrz_ : TransMinusRotatePivotZPlug = PlugDescriptor("transMinusRotatePivotZ")
	node : Transform = None
	pass
class TranslateXPlug(Plug):
	parent : TranslatePlug = PlugDescriptor("translate")
	node : Transform = None
	pass
class TranslateYPlug(Plug):
	parent : TranslatePlug = PlugDescriptor("translate")
	node : Transform = None
	pass
class TranslateZPlug(Plug):
	parent : TranslatePlug = PlugDescriptor("translate")
	node : Transform = None
	pass
class TranslatePlug(Plug):
	translateX_ : TranslateXPlug = PlugDescriptor("translateX")
	tx_ : TranslateXPlug = PlugDescriptor("translateX")
	translateY_ : TranslateYPlug = PlugDescriptor("translateY")
	ty_ : TranslateYPlug = PlugDescriptor("translateY")
	translateZ_ : TranslateZPlug = PlugDescriptor("translateZ")
	tz_ : TranslateZPlug = PlugDescriptor("translateZ")
	node : Transform = None
	pass
class XformMatrixPlug(Plug):
	node : Transform = None
	pass
# endregion


# define node class
class Transform(DagNode):
	dagLocalInverseMatrix_ : DagLocalInverseMatrixPlug = PlugDescriptor("dagLocalInverseMatrix")
	dagLocalMatrix_ : DagLocalMatrixPlug = PlugDescriptor("dagLocalMatrix")
	displayHandle_ : DisplayHandlePlug = PlugDescriptor("displayHandle")
	displayLocalAxis_ : DisplayLocalAxisPlug = PlugDescriptor("displayLocalAxis")
	displayRotatePivot_ : DisplayRotatePivotPlug = PlugDescriptor("displayRotatePivot")
	displayScalePivot_ : DisplayScalePivotPlug = PlugDescriptor("displayScalePivot")
	dynamics_ : DynamicsPlug = PlugDescriptor("dynamics")
	geometry_ : GeometryPlug = PlugDescriptor("geometry")
	inheritsTransform_ : InheritsTransformPlug = PlugDescriptor("inheritsTransform")
	maxRotXLimit_ : MaxRotXLimitPlug = PlugDescriptor("maxRotXLimit")
	maxRotYLimit_ : MaxRotYLimitPlug = PlugDescriptor("maxRotYLimit")
	maxRotZLimit_ : MaxRotZLimitPlug = PlugDescriptor("maxRotZLimit")
	maxRotLimit_ : MaxRotLimitPlug = PlugDescriptor("maxRotLimit")
	maxRotXLimitEnable_ : MaxRotXLimitEnablePlug = PlugDescriptor("maxRotXLimitEnable")
	maxRotYLimitEnable_ : MaxRotYLimitEnablePlug = PlugDescriptor("maxRotYLimitEnable")
	maxRotZLimitEnable_ : MaxRotZLimitEnablePlug = PlugDescriptor("maxRotZLimitEnable")
	maxRotLimitEnable_ : MaxRotLimitEnablePlug = PlugDescriptor("maxRotLimitEnable")
	maxScaleXLimit_ : MaxScaleXLimitPlug = PlugDescriptor("maxScaleXLimit")
	maxScaleYLimit_ : MaxScaleYLimitPlug = PlugDescriptor("maxScaleYLimit")
	maxScaleZLimit_ : MaxScaleZLimitPlug = PlugDescriptor("maxScaleZLimit")
	maxScaleLimit_ : MaxScaleLimitPlug = PlugDescriptor("maxScaleLimit")
	maxScaleXLimitEnable_ : MaxScaleXLimitEnablePlug = PlugDescriptor("maxScaleXLimitEnable")
	maxScaleYLimitEnable_ : MaxScaleYLimitEnablePlug = PlugDescriptor("maxScaleYLimitEnable")
	maxScaleZLimitEnable_ : MaxScaleZLimitEnablePlug = PlugDescriptor("maxScaleZLimitEnable")
	maxScaleLimitEnable_ : MaxScaleLimitEnablePlug = PlugDescriptor("maxScaleLimitEnable")
	maxTransXLimit_ : MaxTransXLimitPlug = PlugDescriptor("maxTransXLimit")
	maxTransYLimit_ : MaxTransYLimitPlug = PlugDescriptor("maxTransYLimit")
	maxTransZLimit_ : MaxTransZLimitPlug = PlugDescriptor("maxTransZLimit")
	maxTransLimit_ : MaxTransLimitPlug = PlugDescriptor("maxTransLimit")
	maxTransXLimitEnable_ : MaxTransXLimitEnablePlug = PlugDescriptor("maxTransXLimitEnable")
	maxTransYLimitEnable_ : MaxTransYLimitEnablePlug = PlugDescriptor("maxTransYLimitEnable")
	maxTransZLimitEnable_ : MaxTransZLimitEnablePlug = PlugDescriptor("maxTransZLimitEnable")
	maxTransLimitEnable_ : MaxTransLimitEnablePlug = PlugDescriptor("maxTransLimitEnable")
	minRotXLimit_ : MinRotXLimitPlug = PlugDescriptor("minRotXLimit")
	minRotYLimit_ : MinRotYLimitPlug = PlugDescriptor("minRotYLimit")
	minRotZLimit_ : MinRotZLimitPlug = PlugDescriptor("minRotZLimit")
	minRotLimit_ : MinRotLimitPlug = PlugDescriptor("minRotLimit")
	minRotXLimitEnable_ : MinRotXLimitEnablePlug = PlugDescriptor("minRotXLimitEnable")
	minRotYLimitEnable_ : MinRotYLimitEnablePlug = PlugDescriptor("minRotYLimitEnable")
	minRotZLimitEnable_ : MinRotZLimitEnablePlug = PlugDescriptor("minRotZLimitEnable")
	minRotLimitEnable_ : MinRotLimitEnablePlug = PlugDescriptor("minRotLimitEnable")
	minScaleXLimit_ : MinScaleXLimitPlug = PlugDescriptor("minScaleXLimit")
	minScaleYLimit_ : MinScaleYLimitPlug = PlugDescriptor("minScaleYLimit")
	minScaleZLimit_ : MinScaleZLimitPlug = PlugDescriptor("minScaleZLimit")
	minScaleLimit_ : MinScaleLimitPlug = PlugDescriptor("minScaleLimit")
	minScaleXLimitEnable_ : MinScaleXLimitEnablePlug = PlugDescriptor("minScaleXLimitEnable")
	minScaleYLimitEnable_ : MinScaleYLimitEnablePlug = PlugDescriptor("minScaleYLimitEnable")
	minScaleZLimitEnable_ : MinScaleZLimitEnablePlug = PlugDescriptor("minScaleZLimitEnable")
	minScaleLimitEnable_ : MinScaleLimitEnablePlug = PlugDescriptor("minScaleLimitEnable")
	minTransXLimit_ : MinTransXLimitPlug = PlugDescriptor("minTransXLimit")
	minTransYLimit_ : MinTransYLimitPlug = PlugDescriptor("minTransYLimit")
	minTransZLimit_ : MinTransZLimitPlug = PlugDescriptor("minTransZLimit")
	minTransLimit_ : MinTransLimitPlug = PlugDescriptor("minTransLimit")
	minTransXLimitEnable_ : MinTransXLimitEnablePlug = PlugDescriptor("minTransXLimitEnable")
	minTransYLimitEnable_ : MinTransYLimitEnablePlug = PlugDescriptor("minTransYLimitEnable")
	minTransZLimitEnable_ : MinTransZLimitEnablePlug = PlugDescriptor("minTransZLimitEnable")
	minTransLimitEnable_ : MinTransLimitEnablePlug = PlugDescriptor("minTransLimitEnable")
	offsetParentMatrix_ : OffsetParentMatrixPlug = PlugDescriptor("offsetParentMatrix")
	rotateX_ : RotateXPlug = PlugDescriptor("rotateX")
	rotateY_ : RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_ : RotateZPlug = PlugDescriptor("rotateZ")
	rotate_ : RotatePlug = PlugDescriptor("rotate")
	rotateAxisX_ : RotateAxisXPlug = PlugDescriptor("rotateAxisX")
	rotateAxisY_ : RotateAxisYPlug = PlugDescriptor("rotateAxisY")
	rotateAxisZ_ : RotateAxisZPlug = PlugDescriptor("rotateAxisZ")
	rotateAxis_ : RotateAxisPlug = PlugDescriptor("rotateAxis")
	rotateOrder_ : RotateOrderPlug = PlugDescriptor("rotateOrder")
	rotatePivotX_ : RotatePivotXPlug = PlugDescriptor("rotatePivotX")
	rotatePivotY_ : RotatePivotYPlug = PlugDescriptor("rotatePivotY")
	rotatePivotZ_ : RotatePivotZPlug = PlugDescriptor("rotatePivotZ")
	rotatePivot_ : RotatePivotPlug = PlugDescriptor("rotatePivot")
	rotatePivotTranslateX_ : RotatePivotTranslateXPlug = PlugDescriptor("rotatePivotTranslateX")
	rotatePivotTranslateY_ : RotatePivotTranslateYPlug = PlugDescriptor("rotatePivotTranslateY")
	rotatePivotTranslateZ_ : RotatePivotTranslateZPlug = PlugDescriptor("rotatePivotTranslateZ")
	rotatePivotTranslate_ : RotatePivotTranslatePlug = PlugDescriptor("rotatePivotTranslate")
	rotateQuaternionW_ : RotateQuaternionWPlug = PlugDescriptor("rotateQuaternionW")
	rotateQuaternionX_ : RotateQuaternionXPlug = PlugDescriptor("rotateQuaternionX")
	rotateQuaternionY_ : RotateQuaternionYPlug = PlugDescriptor("rotateQuaternionY")
	rotateQuaternionZ_ : RotateQuaternionZPlug = PlugDescriptor("rotateQuaternionZ")
	rotateQuaternion_ : RotateQuaternionPlug = PlugDescriptor("rotateQuaternion")
	rotationInterpolation_ : RotationInterpolationPlug = PlugDescriptor("rotationInterpolation")
	scaleX_ : ScaleXPlug = PlugDescriptor("scaleX")
	scaleY_ : ScaleYPlug = PlugDescriptor("scaleY")
	scaleZ_ : ScaleZPlug = PlugDescriptor("scaleZ")
	scale_ : ScalePlug = PlugDescriptor("scale")
	scalePivotX_ : ScalePivotXPlug = PlugDescriptor("scalePivotX")
	scalePivotY_ : ScalePivotYPlug = PlugDescriptor("scalePivotY")
	scalePivotZ_ : ScalePivotZPlug = PlugDescriptor("scalePivotZ")
	scalePivot_ : ScalePivotPlug = PlugDescriptor("scalePivot")
	scalePivotTranslateX_ : ScalePivotTranslateXPlug = PlugDescriptor("scalePivotTranslateX")
	scalePivotTranslateY_ : ScalePivotTranslateYPlug = PlugDescriptor("scalePivotTranslateY")
	scalePivotTranslateZ_ : ScalePivotTranslateZPlug = PlugDescriptor("scalePivotTranslateZ")
	scalePivotTranslate_ : ScalePivotTranslatePlug = PlugDescriptor("scalePivotTranslate")
	selectHandleX_ : SelectHandleXPlug = PlugDescriptor("selectHandleX")
	selectHandleY_ : SelectHandleYPlug = PlugDescriptor("selectHandleY")
	selectHandleZ_ : SelectHandleZPlug = PlugDescriptor("selectHandleZ")
	selectHandle_ : SelectHandlePlug = PlugDescriptor("selectHandle")
	shearXY_ : ShearXYPlug = PlugDescriptor("shearXY")
	shearXZ_ : ShearXZPlug = PlugDescriptor("shearXZ")
	shearYZ_ : ShearYZPlug = PlugDescriptor("shearYZ")
	shear_ : ShearPlug = PlugDescriptor("shear")
	showManipDefault_ : ShowManipDefaultPlug = PlugDescriptor("showManipDefault")
	specifiedManipLocation_ : SpecifiedManipLocationPlug = PlugDescriptor("specifiedManipLocation")
	transMinusRotatePivotX_ : TransMinusRotatePivotXPlug = PlugDescriptor("transMinusRotatePivotX")
	transMinusRotatePivotY_ : TransMinusRotatePivotYPlug = PlugDescriptor("transMinusRotatePivotY")
	transMinusRotatePivotZ_ : TransMinusRotatePivotZPlug = PlugDescriptor("transMinusRotatePivotZ")
	transMinusRotatePivot_ : TransMinusRotatePivotPlug = PlugDescriptor("transMinusRotatePivot")
	translateX_ : TranslateXPlug = PlugDescriptor("translateX")
	translateY_ : TranslateYPlug = PlugDescriptor("translateY")
	translateZ_ : TranslateZPlug = PlugDescriptor("translateZ")
	translate_ : TranslatePlug = PlugDescriptor("translate")
	xformMatrix_ : XformMatrixPlug = PlugDescriptor("xformMatrix")

	# node attributes

	typeName = "transform"
	apiTypeInt = 110
	apiTypeStr = "kTransform"
	typeIdInt = 1481003597
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["dagLocalInverseMatrix", "dagLocalMatrix", "displayHandle", "displayLocalAxis", "displayRotatePivot", "displayScalePivot", "dynamics", "geometry", "inheritsTransform", "maxRotXLimit", "maxRotYLimit", "maxRotZLimit", "maxRotLimit", "maxRotXLimitEnable", "maxRotYLimitEnable", "maxRotZLimitEnable", "maxRotLimitEnable", "maxScaleXLimit", "maxScaleYLimit", "maxScaleZLimit", "maxScaleLimit", "maxScaleXLimitEnable", "maxScaleYLimitEnable", "maxScaleZLimitEnable", "maxScaleLimitEnable", "maxTransXLimit", "maxTransYLimit", "maxTransZLimit", "maxTransLimit", "maxTransXLimitEnable", "maxTransYLimitEnable", "maxTransZLimitEnable", "maxTransLimitEnable", "minRotXLimit", "minRotYLimit", "minRotZLimit", "minRotLimit", "minRotXLimitEnable", "minRotYLimitEnable", "minRotZLimitEnable", "minRotLimitEnable", "minScaleXLimit", "minScaleYLimit", "minScaleZLimit", "minScaleLimit", "minScaleXLimitEnable", "minScaleYLimitEnable", "minScaleZLimitEnable", "minScaleLimitEnable", "minTransXLimit", "minTransYLimit", "minTransZLimit", "minTransLimit", "minTransXLimitEnable", "minTransYLimitEnable", "minTransZLimitEnable", "minTransLimitEnable", "offsetParentMatrix", "rotateX", "rotateY", "rotateZ", "rotate", "rotateAxisX", "rotateAxisY", "rotateAxisZ", "rotateAxis", "rotateOrder", "rotatePivotX", "rotatePivotY", "rotatePivotZ", "rotatePivot", "rotatePivotTranslateX", "rotatePivotTranslateY", "rotatePivotTranslateZ", "rotatePivotTranslate", "rotateQuaternionW", "rotateQuaternionX", "rotateQuaternionY", "rotateQuaternionZ", "rotateQuaternion", "rotationInterpolation", "scaleX", "scaleY", "scaleZ", "scale", "scalePivotX", "scalePivotY", "scalePivotZ", "scalePivot", "scalePivotTranslateX", "scalePivotTranslateY", "scalePivotTranslateZ", "scalePivotTranslate", "selectHandleX", "selectHandleY", "selectHandleZ", "selectHandle", "shearXY", "shearXZ", "shearYZ", "shear", "showManipDefault", "specifiedManipLocation", "transMinusRotatePivotX", "transMinusRotatePivotY", "transMinusRotatePivotZ", "transMinusRotatePivot", "translateX", "translateY", "translateZ", "translate", "xformMatrix"]
	nodeLeafPlugs = ["dagLocalInverseMatrix", "dagLocalMatrix", "displayHandle", "displayLocalAxis", "displayRotatePivot", "displayScalePivot", "dynamics", "geometry", "inheritsTransform", "maxRotLimit", "maxRotLimitEnable", "maxScaleLimit", "maxScaleLimitEnable", "maxTransLimit", "maxTransLimitEnable", "minRotLimit", "minRotLimitEnable", "minScaleLimit", "minScaleLimitEnable", "minTransLimit", "minTransLimitEnable", "offsetParentMatrix", "rotate", "rotateAxis", "rotateOrder", "rotatePivot", "rotatePivotTranslate", "rotateQuaternion", "rotationInterpolation", "scale", "scalePivot", "scalePivotTranslate", "selectHandle", "shear", "showManipDefault", "specifiedManipLocation", "transMinusRotatePivot", "translate", "xformMatrix"]
	pass

