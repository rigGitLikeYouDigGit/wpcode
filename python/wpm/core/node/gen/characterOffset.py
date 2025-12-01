

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
class ApplyControlParentTransformPlug(Plug):
	node : CharacterOffset = None
	pass
class BinMembershipPlug(Plug):
	node : CharacterOffset = None
	pass
class EnablePlug(Plug):
	node : CharacterOffset = None
	pass
class InRootRotateXPlug(Plug):
	parent : InRootRotatePlug = PlugDescriptor("inRootRotate")
	node : CharacterOffset = None
	pass
class InRootRotateYPlug(Plug):
	parent : InRootRotatePlug = PlugDescriptor("inRootRotate")
	node : CharacterOffset = None
	pass
class InRootRotateZPlug(Plug):
	parent : InRootRotatePlug = PlugDescriptor("inRootRotate")
	node : CharacterOffset = None
	pass
class InRootRotatePlug(Plug):
	inRootRotateX_ : InRootRotateXPlug = PlugDescriptor("inRootRotateX")
	rrix_ : InRootRotateXPlug = PlugDescriptor("inRootRotateX")
	inRootRotateY_ : InRootRotateYPlug = PlugDescriptor("inRootRotateY")
	rriy_ : InRootRotateYPlug = PlugDescriptor("inRootRotateY")
	inRootRotateZ_ : InRootRotateZPlug = PlugDescriptor("inRootRotateZ")
	rriz_ : InRootRotateZPlug = PlugDescriptor("inRootRotateZ")
	node : CharacterOffset = None
	pass
class InRootTranslateXPlug(Plug):
	parent : InRootTranslatePlug = PlugDescriptor("inRootTranslate")
	node : CharacterOffset = None
	pass
class InRootTranslateYPlug(Plug):
	parent : InRootTranslatePlug = PlugDescriptor("inRootTranslate")
	node : CharacterOffset = None
	pass
class InRootTranslateZPlug(Plug):
	parent : InRootTranslatePlug = PlugDescriptor("inRootTranslate")
	node : CharacterOffset = None
	pass
class InRootTranslatePlug(Plug):
	inRootTranslateX_ : InRootTranslateXPlug = PlugDescriptor("inRootTranslateX")
	rtix_ : InRootTranslateXPlug = PlugDescriptor("inRootTranslateX")
	inRootTranslateY_ : InRootTranslateYPlug = PlugDescriptor("inRootTranslateY")
	rtiy_ : InRootTranslateYPlug = PlugDescriptor("inRootTranslateY")
	inRootTranslateZ_ : InRootTranslateZPlug = PlugDescriptor("inRootTranslateZ")
	rtiz_ : InRootTranslateZPlug = PlugDescriptor("inRootTranslateZ")
	node : CharacterOffset = None
	pass
class InitialOffsetRootTranslateXPlug(Plug):
	parent : InitialOffsetRootTranslatePlug = PlugDescriptor("initialOffsetRootTranslate")
	node : CharacterOffset = None
	pass
class InitialOffsetRootTranslateYPlug(Plug):
	parent : InitialOffsetRootTranslatePlug = PlugDescriptor("initialOffsetRootTranslate")
	node : CharacterOffset = None
	pass
class InitialOffsetRootTranslateZPlug(Plug):
	parent : InitialOffsetRootTranslatePlug = PlugDescriptor("initialOffsetRootTranslate")
	node : CharacterOffset = None
	pass
class InitialOffsetRootTranslatePlug(Plug):
	initialOffsetRootTranslateX_ : InitialOffsetRootTranslateXPlug = PlugDescriptor("initialOffsetRootTranslateX")
	itfx_ : InitialOffsetRootTranslateXPlug = PlugDescriptor("initialOffsetRootTranslateX")
	initialOffsetRootTranslateY_ : InitialOffsetRootTranslateYPlug = PlugDescriptor("initialOffsetRootTranslateY")
	itfy_ : InitialOffsetRootTranslateYPlug = PlugDescriptor("initialOffsetRootTranslateY")
	initialOffsetRootTranslateZ_ : InitialOffsetRootTranslateZPlug = PlugDescriptor("initialOffsetRootTranslateZ")
	itfz_ : InitialOffsetRootTranslateZPlug = PlugDescriptor("initialOffsetRootTranslateZ")
	node : CharacterOffset = None
	pass
class OffsetRootRotateXPlug(Plug):
	parent : OffsetRootRotatePlug = PlugDescriptor("offsetRootRotate")
	node : CharacterOffset = None
	pass
class OffsetRootRotateYPlug(Plug):
	parent : OffsetRootRotatePlug = PlugDescriptor("offsetRootRotate")
	node : CharacterOffset = None
	pass
class OffsetRootRotateZPlug(Plug):
	parent : OffsetRootRotatePlug = PlugDescriptor("offsetRootRotate")
	node : CharacterOffset = None
	pass
class OffsetRootRotatePlug(Plug):
	offsetRootRotateX_ : OffsetRootRotateXPlug = PlugDescriptor("offsetRootRotateX")
	rrfx_ : OffsetRootRotateXPlug = PlugDescriptor("offsetRootRotateX")
	offsetRootRotateY_ : OffsetRootRotateYPlug = PlugDescriptor("offsetRootRotateY")
	rrfy_ : OffsetRootRotateYPlug = PlugDescriptor("offsetRootRotateY")
	offsetRootRotateZ_ : OffsetRootRotateZPlug = PlugDescriptor("offsetRootRotateZ")
	rrfz_ : OffsetRootRotateZPlug = PlugDescriptor("offsetRootRotateZ")
	node : CharacterOffset = None
	pass
class OffsetRootRotateOrderPlug(Plug):
	node : CharacterOffset = None
	pass
class OffsetRootRotatePivotXPlug(Plug):
	parent : OffsetRootRotatePivotPlug = PlugDescriptor("offsetRootRotatePivot")
	node : CharacterOffset = None
	pass
class OffsetRootRotatePivotYPlug(Plug):
	parent : OffsetRootRotatePivotPlug = PlugDescriptor("offsetRootRotatePivot")
	node : CharacterOffset = None
	pass
class OffsetRootRotatePivotZPlug(Plug):
	parent : OffsetRootRotatePivotPlug = PlugDescriptor("offsetRootRotatePivot")
	node : CharacterOffset = None
	pass
class OffsetRootRotatePivotPlug(Plug):
	offsetRootRotatePivotX_ : OffsetRootRotatePivotXPlug = PlugDescriptor("offsetRootRotatePivotX")
	rppfx_ : OffsetRootRotatePivotXPlug = PlugDescriptor("offsetRootRotatePivotX")
	offsetRootRotatePivotY_ : OffsetRootRotatePivotYPlug = PlugDescriptor("offsetRootRotatePivotY")
	rppfy_ : OffsetRootRotatePivotYPlug = PlugDescriptor("offsetRootRotatePivotY")
	offsetRootRotatePivotZ_ : OffsetRootRotatePivotZPlug = PlugDescriptor("offsetRootRotatePivotZ")
	rppfz_ : OffsetRootRotatePivotZPlug = PlugDescriptor("offsetRootRotatePivotZ")
	node : CharacterOffset = None
	pass
class OffsetRootTranslateXPlug(Plug):
	parent : OffsetRootTranslatePlug = PlugDescriptor("offsetRootTranslate")
	node : CharacterOffset = None
	pass
class OffsetRootTranslateYPlug(Plug):
	parent : OffsetRootTranslatePlug = PlugDescriptor("offsetRootTranslate")
	node : CharacterOffset = None
	pass
class OffsetRootTranslateZPlug(Plug):
	parent : OffsetRootTranslatePlug = PlugDescriptor("offsetRootTranslate")
	node : CharacterOffset = None
	pass
class OffsetRootTranslatePlug(Plug):
	offsetRootTranslateX_ : OffsetRootTranslateXPlug = PlugDescriptor("offsetRootTranslateX")
	rtfx_ : OffsetRootTranslateXPlug = PlugDescriptor("offsetRootTranslateX")
	offsetRootTranslateY_ : OffsetRootTranslateYPlug = PlugDescriptor("offsetRootTranslateY")
	rtfy_ : OffsetRootTranslateYPlug = PlugDescriptor("offsetRootTranslateY")
	offsetRootTranslateZ_ : OffsetRootTranslateZPlug = PlugDescriptor("offsetRootTranslateZ")
	rtfz_ : OffsetRootTranslateZPlug = PlugDescriptor("offsetRootTranslateZ")
	node : CharacterOffset = None
	pass
class OutRootRotateXPlug(Plug):
	parent : OutRootRotatePlug = PlugDescriptor("outRootRotate")
	node : CharacterOffset = None
	pass
class OutRootRotateYPlug(Plug):
	parent : OutRootRotatePlug = PlugDescriptor("outRootRotate")
	node : CharacterOffset = None
	pass
class OutRootRotateZPlug(Plug):
	parent : OutRootRotatePlug = PlugDescriptor("outRootRotate")
	node : CharacterOffset = None
	pass
class OutRootRotatePlug(Plug):
	outRootRotateX_ : OutRootRotateXPlug = PlugDescriptor("outRootRotateX")
	rrox_ : OutRootRotateXPlug = PlugDescriptor("outRootRotateX")
	outRootRotateY_ : OutRootRotateYPlug = PlugDescriptor("outRootRotateY")
	rroy_ : OutRootRotateYPlug = PlugDescriptor("outRootRotateY")
	outRootRotateZ_ : OutRootRotateZPlug = PlugDescriptor("outRootRotateZ")
	rroz_ : OutRootRotateZPlug = PlugDescriptor("outRootRotateZ")
	node : CharacterOffset = None
	pass
class OutRootTranslateXPlug(Plug):
	parent : OutRootTranslatePlug = PlugDescriptor("outRootTranslate")
	node : CharacterOffset = None
	pass
class OutRootTranslateYPlug(Plug):
	parent : OutRootTranslatePlug = PlugDescriptor("outRootTranslate")
	node : CharacterOffset = None
	pass
class OutRootTranslateZPlug(Plug):
	parent : OutRootTranslatePlug = PlugDescriptor("outRootTranslate")
	node : CharacterOffset = None
	pass
class OutRootTranslatePlug(Plug):
	outRootTranslateX_ : OutRootTranslateXPlug = PlugDescriptor("outRootTranslateX")
	rtox_ : OutRootTranslateXPlug = PlugDescriptor("outRootTranslateX")
	outRootTranslateY_ : OutRootTranslateYPlug = PlugDescriptor("outRootTranslateY")
	rtoy_ : OutRootTranslateYPlug = PlugDescriptor("outRootTranslateY")
	outRootTranslateZ_ : OutRootTranslateZPlug = PlugDescriptor("outRootTranslateZ")
	rtoz_ : OutRootTranslateZPlug = PlugDescriptor("outRootTranslateZ")
	node : CharacterOffset = None
	pass
class RootJointOrientXPlug(Plug):
	parent : RootJointOrientPlug = PlugDescriptor("rootJointOrient")
	node : CharacterOffset = None
	pass
class RootJointOrientYPlug(Plug):
	parent : RootJointOrientPlug = PlugDescriptor("rootJointOrient")
	node : CharacterOffset = None
	pass
class RootJointOrientZPlug(Plug):
	parent : RootJointOrientPlug = PlugDescriptor("rootJointOrient")
	node : CharacterOffset = None
	pass
class RootJointOrientPlug(Plug):
	rootJointOrientX_ : RootJointOrientXPlug = PlugDescriptor("rootJointOrientX")
	rjox_ : RootJointOrientXPlug = PlugDescriptor("rootJointOrientX")
	rootJointOrientY_ : RootJointOrientYPlug = PlugDescriptor("rootJointOrientY")
	rjoy_ : RootJointOrientYPlug = PlugDescriptor("rootJointOrientY")
	rootJointOrientZ_ : RootJointOrientZPlug = PlugDescriptor("rootJointOrientZ")
	rjoz_ : RootJointOrientZPlug = PlugDescriptor("rootJointOrientZ")
	node : CharacterOffset = None
	pass
class RootParentInverseMatrixPlug(Plug):
	node : CharacterOffset = None
	pass
class RootParentMatrixPlug(Plug):
	node : CharacterOffset = None
	pass
class RootRotateOrderPlug(Plug):
	node : CharacterOffset = None
	pass
class RotateControlParentMatrixPlug(Plug):
	node : CharacterOffset = None
	pass
class RotateControlScaleXPlug(Plug):
	parent : RotateControlScalePlug = PlugDescriptor("rotateControlScale")
	node : CharacterOffset = None
	pass
class RotateControlScaleYPlug(Plug):
	parent : RotateControlScalePlug = PlugDescriptor("rotateControlScale")
	node : CharacterOffset = None
	pass
class RotateControlScaleZPlug(Plug):
	parent : RotateControlScalePlug = PlugDescriptor("rotateControlScale")
	node : CharacterOffset = None
	pass
class RotateControlScalePlug(Plug):
	rotateControlScaleX_ : RotateControlScaleXPlug = PlugDescriptor("rotateControlScaleX")
	rcsx_ : RotateControlScaleXPlug = PlugDescriptor("rotateControlScaleX")
	rotateControlScaleY_ : RotateControlScaleYPlug = PlugDescriptor("rotateControlScaleY")
	rcsy_ : RotateControlScaleYPlug = PlugDescriptor("rotateControlScaleY")
	rotateControlScaleZ_ : RotateControlScaleZPlug = PlugDescriptor("rotateControlScaleZ")
	rcsz_ : RotateControlScaleZPlug = PlugDescriptor("rotateControlScaleZ")
	node : CharacterOffset = None
	pass
# endregion


# define node class
class CharacterOffset(_BASE_):
	applyControlParentTransform_ : ApplyControlParentTransformPlug = PlugDescriptor("applyControlParentTransform")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	enable_ : EnablePlug = PlugDescriptor("enable")
	inRootRotateX_ : InRootRotateXPlug = PlugDescriptor("inRootRotateX")
	inRootRotateY_ : InRootRotateYPlug = PlugDescriptor("inRootRotateY")
	inRootRotateZ_ : InRootRotateZPlug = PlugDescriptor("inRootRotateZ")
	inRootRotate_ : InRootRotatePlug = PlugDescriptor("inRootRotate")
	inRootTranslateX_ : InRootTranslateXPlug = PlugDescriptor("inRootTranslateX")
	inRootTranslateY_ : InRootTranslateYPlug = PlugDescriptor("inRootTranslateY")
	inRootTranslateZ_ : InRootTranslateZPlug = PlugDescriptor("inRootTranslateZ")
	inRootTranslate_ : InRootTranslatePlug = PlugDescriptor("inRootTranslate")
	initialOffsetRootTranslateX_ : InitialOffsetRootTranslateXPlug = PlugDescriptor("initialOffsetRootTranslateX")
	initialOffsetRootTranslateY_ : InitialOffsetRootTranslateYPlug = PlugDescriptor("initialOffsetRootTranslateY")
	initialOffsetRootTranslateZ_ : InitialOffsetRootTranslateZPlug = PlugDescriptor("initialOffsetRootTranslateZ")
	initialOffsetRootTranslate_ : InitialOffsetRootTranslatePlug = PlugDescriptor("initialOffsetRootTranslate")
	offsetRootRotateX_ : OffsetRootRotateXPlug = PlugDescriptor("offsetRootRotateX")
	offsetRootRotateY_ : OffsetRootRotateYPlug = PlugDescriptor("offsetRootRotateY")
	offsetRootRotateZ_ : OffsetRootRotateZPlug = PlugDescriptor("offsetRootRotateZ")
	offsetRootRotate_ : OffsetRootRotatePlug = PlugDescriptor("offsetRootRotate")
	offsetRootRotateOrder_ : OffsetRootRotateOrderPlug = PlugDescriptor("offsetRootRotateOrder")
	offsetRootRotatePivotX_ : OffsetRootRotatePivotXPlug = PlugDescriptor("offsetRootRotatePivotX")
	offsetRootRotatePivotY_ : OffsetRootRotatePivotYPlug = PlugDescriptor("offsetRootRotatePivotY")
	offsetRootRotatePivotZ_ : OffsetRootRotatePivotZPlug = PlugDescriptor("offsetRootRotatePivotZ")
	offsetRootRotatePivot_ : OffsetRootRotatePivotPlug = PlugDescriptor("offsetRootRotatePivot")
	offsetRootTranslateX_ : OffsetRootTranslateXPlug = PlugDescriptor("offsetRootTranslateX")
	offsetRootTranslateY_ : OffsetRootTranslateYPlug = PlugDescriptor("offsetRootTranslateY")
	offsetRootTranslateZ_ : OffsetRootTranslateZPlug = PlugDescriptor("offsetRootTranslateZ")
	offsetRootTranslate_ : OffsetRootTranslatePlug = PlugDescriptor("offsetRootTranslate")
	outRootRotateX_ : OutRootRotateXPlug = PlugDescriptor("outRootRotateX")
	outRootRotateY_ : OutRootRotateYPlug = PlugDescriptor("outRootRotateY")
	outRootRotateZ_ : OutRootRotateZPlug = PlugDescriptor("outRootRotateZ")
	outRootRotate_ : OutRootRotatePlug = PlugDescriptor("outRootRotate")
	outRootTranslateX_ : OutRootTranslateXPlug = PlugDescriptor("outRootTranslateX")
	outRootTranslateY_ : OutRootTranslateYPlug = PlugDescriptor("outRootTranslateY")
	outRootTranslateZ_ : OutRootTranslateZPlug = PlugDescriptor("outRootTranslateZ")
	outRootTranslate_ : OutRootTranslatePlug = PlugDescriptor("outRootTranslate")
	rootJointOrientX_ : RootJointOrientXPlug = PlugDescriptor("rootJointOrientX")
	rootJointOrientY_ : RootJointOrientYPlug = PlugDescriptor("rootJointOrientY")
	rootJointOrientZ_ : RootJointOrientZPlug = PlugDescriptor("rootJointOrientZ")
	rootJointOrient_ : RootJointOrientPlug = PlugDescriptor("rootJointOrient")
	rootParentInverseMatrix_ : RootParentInverseMatrixPlug = PlugDescriptor("rootParentInverseMatrix")
	rootParentMatrix_ : RootParentMatrixPlug = PlugDescriptor("rootParentMatrix")
	rootRotateOrder_ : RootRotateOrderPlug = PlugDescriptor("rootRotateOrder")
	rotateControlParentMatrix_ : RotateControlParentMatrixPlug = PlugDescriptor("rotateControlParentMatrix")
	rotateControlScaleX_ : RotateControlScaleXPlug = PlugDescriptor("rotateControlScaleX")
	rotateControlScaleY_ : RotateControlScaleYPlug = PlugDescriptor("rotateControlScaleY")
	rotateControlScaleZ_ : RotateControlScaleZPlug = PlugDescriptor("rotateControlScaleZ")
	rotateControlScale_ : RotateControlScalePlug = PlugDescriptor("rotateControlScale")

	# node attributes

	typeName = "characterOffset"
	apiTypeInt = 689
	apiTypeStr = "kCharacterOffset"
	typeIdInt = 1481590342
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["applyControlParentTransform", "binMembership", "enable", "inRootRotateX", "inRootRotateY", "inRootRotateZ", "inRootRotate", "inRootTranslateX", "inRootTranslateY", "inRootTranslateZ", "inRootTranslate", "initialOffsetRootTranslateX", "initialOffsetRootTranslateY", "initialOffsetRootTranslateZ", "initialOffsetRootTranslate", "offsetRootRotateX", "offsetRootRotateY", "offsetRootRotateZ", "offsetRootRotate", "offsetRootRotateOrder", "offsetRootRotatePivotX", "offsetRootRotatePivotY", "offsetRootRotatePivotZ", "offsetRootRotatePivot", "offsetRootTranslateX", "offsetRootTranslateY", "offsetRootTranslateZ", "offsetRootTranslate", "outRootRotateX", "outRootRotateY", "outRootRotateZ", "outRootRotate", "outRootTranslateX", "outRootTranslateY", "outRootTranslateZ", "outRootTranslate", "rootJointOrientX", "rootJointOrientY", "rootJointOrientZ", "rootJointOrient", "rootParentInverseMatrix", "rootParentMatrix", "rootRotateOrder", "rotateControlParentMatrix", "rotateControlScaleX", "rotateControlScaleY", "rotateControlScaleZ", "rotateControlScale"]
	nodeLeafPlugs = ["applyControlParentTransform", "binMembership", "enable", "inRootRotate", "inRootTranslate", "initialOffsetRootTranslate", "offsetRootRotate", "offsetRootRotateOrder", "offsetRootRotatePivot", "offsetRootTranslate", "outRootRotate", "outRootTranslate", "rootJointOrient", "rootParentInverseMatrix", "rootParentMatrix", "rootRotateOrder", "rotateControlParentMatrix", "rotateControlScale"]
	pass

