

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
	node : ComposeMatrix = None
	pass
class InputQuatWPlug(Plug):
	parent : InputQuatPlug = PlugDescriptor("inputQuat")
	node : ComposeMatrix = None
	pass
class InputQuatXPlug(Plug):
	parent : InputQuatPlug = PlugDescriptor("inputQuat")
	node : ComposeMatrix = None
	pass
class InputQuatYPlug(Plug):
	parent : InputQuatPlug = PlugDescriptor("inputQuat")
	node : ComposeMatrix = None
	pass
class InputQuatZPlug(Plug):
	parent : InputQuatPlug = PlugDescriptor("inputQuat")
	node : ComposeMatrix = None
	pass
class InputQuatPlug(Plug):
	inputQuatW_ : InputQuatWPlug = PlugDescriptor("inputQuatW")
	iqw_ : InputQuatWPlug = PlugDescriptor("inputQuatW")
	inputQuatX_ : InputQuatXPlug = PlugDescriptor("inputQuatX")
	iqwx_ : InputQuatXPlug = PlugDescriptor("inputQuatX")
	inputQuatY_ : InputQuatYPlug = PlugDescriptor("inputQuatY")
	iqwy_ : InputQuatYPlug = PlugDescriptor("inputQuatY")
	inputQuatZ_ : InputQuatZPlug = PlugDescriptor("inputQuatZ")
	iqwz_ : InputQuatZPlug = PlugDescriptor("inputQuatZ")
	node : ComposeMatrix = None
	pass
class InputRotateXPlug(Plug):
	parent : InputRotatePlug = PlugDescriptor("inputRotate")
	node : ComposeMatrix = None
	pass
class InputRotateYPlug(Plug):
	parent : InputRotatePlug = PlugDescriptor("inputRotate")
	node : ComposeMatrix = None
	pass
class InputRotateZPlug(Plug):
	parent : InputRotatePlug = PlugDescriptor("inputRotate")
	node : ComposeMatrix = None
	pass
class InputRotatePlug(Plug):
	inputRotateX_ : InputRotateXPlug = PlugDescriptor("inputRotateX")
	irx_ : InputRotateXPlug = PlugDescriptor("inputRotateX")
	inputRotateY_ : InputRotateYPlug = PlugDescriptor("inputRotateY")
	iry_ : InputRotateYPlug = PlugDescriptor("inputRotateY")
	inputRotateZ_ : InputRotateZPlug = PlugDescriptor("inputRotateZ")
	irz_ : InputRotateZPlug = PlugDescriptor("inputRotateZ")
	node : ComposeMatrix = None
	pass
class InputRotateOrderPlug(Plug):
	node : ComposeMatrix = None
	pass
class InputScaleXPlug(Plug):
	parent : InputScalePlug = PlugDescriptor("inputScale")
	node : ComposeMatrix = None
	pass
class InputScaleYPlug(Plug):
	parent : InputScalePlug = PlugDescriptor("inputScale")
	node : ComposeMatrix = None
	pass
class InputScaleZPlug(Plug):
	parent : InputScalePlug = PlugDescriptor("inputScale")
	node : ComposeMatrix = None
	pass
class InputScalePlug(Plug):
	inputScaleX_ : InputScaleXPlug = PlugDescriptor("inputScaleX")
	isx_ : InputScaleXPlug = PlugDescriptor("inputScaleX")
	inputScaleY_ : InputScaleYPlug = PlugDescriptor("inputScaleY")
	isy_ : InputScaleYPlug = PlugDescriptor("inputScaleY")
	inputScaleZ_ : InputScaleZPlug = PlugDescriptor("inputScaleZ")
	isz_ : InputScaleZPlug = PlugDescriptor("inputScaleZ")
	node : ComposeMatrix = None
	pass
class InputShearXPlug(Plug):
	parent : InputShearPlug = PlugDescriptor("inputShear")
	node : ComposeMatrix = None
	pass
class InputShearYPlug(Plug):
	parent : InputShearPlug = PlugDescriptor("inputShear")
	node : ComposeMatrix = None
	pass
class InputShearZPlug(Plug):
	parent : InputShearPlug = PlugDescriptor("inputShear")
	node : ComposeMatrix = None
	pass
class InputShearPlug(Plug):
	inputShearX_ : InputShearXPlug = PlugDescriptor("inputShearX")
	ishx_ : InputShearXPlug = PlugDescriptor("inputShearX")
	inputShearY_ : InputShearYPlug = PlugDescriptor("inputShearY")
	ishy_ : InputShearYPlug = PlugDescriptor("inputShearY")
	inputShearZ_ : InputShearZPlug = PlugDescriptor("inputShearZ")
	ishz_ : InputShearZPlug = PlugDescriptor("inputShearZ")
	node : ComposeMatrix = None
	pass
class InputTranslateXPlug(Plug):
	parent : InputTranslatePlug = PlugDescriptor("inputTranslate")
	node : ComposeMatrix = None
	pass
class InputTranslateYPlug(Plug):
	parent : InputTranslatePlug = PlugDescriptor("inputTranslate")
	node : ComposeMatrix = None
	pass
class InputTranslateZPlug(Plug):
	parent : InputTranslatePlug = PlugDescriptor("inputTranslate")
	node : ComposeMatrix = None
	pass
class InputTranslatePlug(Plug):
	inputTranslateX_ : InputTranslateXPlug = PlugDescriptor("inputTranslateX")
	itx_ : InputTranslateXPlug = PlugDescriptor("inputTranslateX")
	inputTranslateY_ : InputTranslateYPlug = PlugDescriptor("inputTranslateY")
	ity_ : InputTranslateYPlug = PlugDescriptor("inputTranslateY")
	inputTranslateZ_ : InputTranslateZPlug = PlugDescriptor("inputTranslateZ")
	itz_ : InputTranslateZPlug = PlugDescriptor("inputTranslateZ")
	node : ComposeMatrix = None
	pass
class OutputMatrixPlug(Plug):
	node : ComposeMatrix = None
	pass
class UseEulerRotationPlug(Plug):
	node : ComposeMatrix = None
	pass
# endregion


# define node class
class ComposeMatrix(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	inputQuatW_ : InputQuatWPlug = PlugDescriptor("inputQuatW")
	inputQuatX_ : InputQuatXPlug = PlugDescriptor("inputQuatX")
	inputQuatY_ : InputQuatYPlug = PlugDescriptor("inputQuatY")
	inputQuatZ_ : InputQuatZPlug = PlugDescriptor("inputQuatZ")
	inputQuat_ : InputQuatPlug = PlugDescriptor("inputQuat")
	inputRotateX_ : InputRotateXPlug = PlugDescriptor("inputRotateX")
	inputRotateY_ : InputRotateYPlug = PlugDescriptor("inputRotateY")
	inputRotateZ_ : InputRotateZPlug = PlugDescriptor("inputRotateZ")
	inputRotate_ : InputRotatePlug = PlugDescriptor("inputRotate")
	inputRotateOrder_ : InputRotateOrderPlug = PlugDescriptor("inputRotateOrder")
	inputScaleX_ : InputScaleXPlug = PlugDescriptor("inputScaleX")
	inputScaleY_ : InputScaleYPlug = PlugDescriptor("inputScaleY")
	inputScaleZ_ : InputScaleZPlug = PlugDescriptor("inputScaleZ")
	inputScale_ : InputScalePlug = PlugDescriptor("inputScale")
	inputShearX_ : InputShearXPlug = PlugDescriptor("inputShearX")
	inputShearY_ : InputShearYPlug = PlugDescriptor("inputShearY")
	inputShearZ_ : InputShearZPlug = PlugDescriptor("inputShearZ")
	inputShear_ : InputShearPlug = PlugDescriptor("inputShear")
	inputTranslateX_ : InputTranslateXPlug = PlugDescriptor("inputTranslateX")
	inputTranslateY_ : InputTranslateYPlug = PlugDescriptor("inputTranslateY")
	inputTranslateZ_ : InputTranslateZPlug = PlugDescriptor("inputTranslateZ")
	inputTranslate_ : InputTranslatePlug = PlugDescriptor("inputTranslate")
	outputMatrix_ : OutputMatrixPlug = PlugDescriptor("outputMatrix")
	useEulerRotation_ : UseEulerRotationPlug = PlugDescriptor("useEulerRotation")

	# node attributes

	typeName = "composeMatrix"
	apiTypeInt = 1136
	apiTypeStr = "kComposeMatrix"
	typeIdInt = 1476395777
	MFnCls = om.MFnDependencyNode
	pass

