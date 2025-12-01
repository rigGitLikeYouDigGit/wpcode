

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
	node : DecomposeMatrix = None
	pass
class InputMatrixPlug(Plug):
	node : DecomposeMatrix = None
	pass
class InputRotateOrderPlug(Plug):
	node : DecomposeMatrix = None
	pass
class OutputQuatWPlug(Plug):
	parent : OutputQuatPlug = PlugDescriptor("outputQuat")
	node : DecomposeMatrix = None
	pass
class OutputQuatXPlug(Plug):
	parent : OutputQuatPlug = PlugDescriptor("outputQuat")
	node : DecomposeMatrix = None
	pass
class OutputQuatYPlug(Plug):
	parent : OutputQuatPlug = PlugDescriptor("outputQuat")
	node : DecomposeMatrix = None
	pass
class OutputQuatZPlug(Plug):
	parent : OutputQuatPlug = PlugDescriptor("outputQuat")
	node : DecomposeMatrix = None
	pass
class OutputQuatPlug(Plug):
	outputQuatW_ : OutputQuatWPlug = PlugDescriptor("outputQuatW")
	oqw_ : OutputQuatWPlug = PlugDescriptor("outputQuatW")
	outputQuatX_ : OutputQuatXPlug = PlugDescriptor("outputQuatX")
	oqx_ : OutputQuatXPlug = PlugDescriptor("outputQuatX")
	outputQuatY_ : OutputQuatYPlug = PlugDescriptor("outputQuatY")
	oqy_ : OutputQuatYPlug = PlugDescriptor("outputQuatY")
	outputQuatZ_ : OutputQuatZPlug = PlugDescriptor("outputQuatZ")
	oqz_ : OutputQuatZPlug = PlugDescriptor("outputQuatZ")
	node : DecomposeMatrix = None
	pass
class OutputRotateXPlug(Plug):
	parent : OutputRotatePlug = PlugDescriptor("outputRotate")
	node : DecomposeMatrix = None
	pass
class OutputRotateYPlug(Plug):
	parent : OutputRotatePlug = PlugDescriptor("outputRotate")
	node : DecomposeMatrix = None
	pass
class OutputRotateZPlug(Plug):
	parent : OutputRotatePlug = PlugDescriptor("outputRotate")
	node : DecomposeMatrix = None
	pass
class OutputRotatePlug(Plug):
	outputRotateX_ : OutputRotateXPlug = PlugDescriptor("outputRotateX")
	orx_ : OutputRotateXPlug = PlugDescriptor("outputRotateX")
	outputRotateY_ : OutputRotateYPlug = PlugDescriptor("outputRotateY")
	ory_ : OutputRotateYPlug = PlugDescriptor("outputRotateY")
	outputRotateZ_ : OutputRotateZPlug = PlugDescriptor("outputRotateZ")
	orz_ : OutputRotateZPlug = PlugDescriptor("outputRotateZ")
	node : DecomposeMatrix = None
	pass
class OutputScaleXPlug(Plug):
	parent : OutputScalePlug = PlugDescriptor("outputScale")
	node : DecomposeMatrix = None
	pass
class OutputScaleYPlug(Plug):
	parent : OutputScalePlug = PlugDescriptor("outputScale")
	node : DecomposeMatrix = None
	pass
class OutputScaleZPlug(Plug):
	parent : OutputScalePlug = PlugDescriptor("outputScale")
	node : DecomposeMatrix = None
	pass
class OutputScalePlug(Plug):
	outputScaleX_ : OutputScaleXPlug = PlugDescriptor("outputScaleX")
	osx_ : OutputScaleXPlug = PlugDescriptor("outputScaleX")
	outputScaleY_ : OutputScaleYPlug = PlugDescriptor("outputScaleY")
	osy_ : OutputScaleYPlug = PlugDescriptor("outputScaleY")
	outputScaleZ_ : OutputScaleZPlug = PlugDescriptor("outputScaleZ")
	osz_ : OutputScaleZPlug = PlugDescriptor("outputScaleZ")
	node : DecomposeMatrix = None
	pass
class OutputShearXPlug(Plug):
	parent : OutputShearPlug = PlugDescriptor("outputShear")
	node : DecomposeMatrix = None
	pass
class OutputShearYPlug(Plug):
	parent : OutputShearPlug = PlugDescriptor("outputShear")
	node : DecomposeMatrix = None
	pass
class OutputShearZPlug(Plug):
	parent : OutputShearPlug = PlugDescriptor("outputShear")
	node : DecomposeMatrix = None
	pass
class OutputShearPlug(Plug):
	outputShearX_ : OutputShearXPlug = PlugDescriptor("outputShearX")
	oshx_ : OutputShearXPlug = PlugDescriptor("outputShearX")
	outputShearY_ : OutputShearYPlug = PlugDescriptor("outputShearY")
	oshy_ : OutputShearYPlug = PlugDescriptor("outputShearY")
	outputShearZ_ : OutputShearZPlug = PlugDescriptor("outputShearZ")
	oshz_ : OutputShearZPlug = PlugDescriptor("outputShearZ")
	node : DecomposeMatrix = None
	pass
class OutputTranslateXPlug(Plug):
	parent : OutputTranslatePlug = PlugDescriptor("outputTranslate")
	node : DecomposeMatrix = None
	pass
class OutputTranslateYPlug(Plug):
	parent : OutputTranslatePlug = PlugDescriptor("outputTranslate")
	node : DecomposeMatrix = None
	pass
class OutputTranslateZPlug(Plug):
	parent : OutputTranslatePlug = PlugDescriptor("outputTranslate")
	node : DecomposeMatrix = None
	pass
class OutputTranslatePlug(Plug):
	outputTranslateX_ : OutputTranslateXPlug = PlugDescriptor("outputTranslateX")
	otx_ : OutputTranslateXPlug = PlugDescriptor("outputTranslateX")
	outputTranslateY_ : OutputTranslateYPlug = PlugDescriptor("outputTranslateY")
	oty_ : OutputTranslateYPlug = PlugDescriptor("outputTranslateY")
	outputTranslateZ_ : OutputTranslateZPlug = PlugDescriptor("outputTranslateZ")
	otz_ : OutputTranslateZPlug = PlugDescriptor("outputTranslateZ")
	node : DecomposeMatrix = None
	pass
# endregion


# define node class
class DecomposeMatrix(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	inputMatrix_ : InputMatrixPlug = PlugDescriptor("inputMatrix")
	inputRotateOrder_ : InputRotateOrderPlug = PlugDescriptor("inputRotateOrder")
	outputQuatW_ : OutputQuatWPlug = PlugDescriptor("outputQuatW")
	outputQuatX_ : OutputQuatXPlug = PlugDescriptor("outputQuatX")
	outputQuatY_ : OutputQuatYPlug = PlugDescriptor("outputQuatY")
	outputQuatZ_ : OutputQuatZPlug = PlugDescriptor("outputQuatZ")
	outputQuat_ : OutputQuatPlug = PlugDescriptor("outputQuat")
	outputRotateX_ : OutputRotateXPlug = PlugDescriptor("outputRotateX")
	outputRotateY_ : OutputRotateYPlug = PlugDescriptor("outputRotateY")
	outputRotateZ_ : OutputRotateZPlug = PlugDescriptor("outputRotateZ")
	outputRotate_ : OutputRotatePlug = PlugDescriptor("outputRotate")
	outputScaleX_ : OutputScaleXPlug = PlugDescriptor("outputScaleX")
	outputScaleY_ : OutputScaleYPlug = PlugDescriptor("outputScaleY")
	outputScaleZ_ : OutputScaleZPlug = PlugDescriptor("outputScaleZ")
	outputScale_ : OutputScalePlug = PlugDescriptor("outputScale")
	outputShearX_ : OutputShearXPlug = PlugDescriptor("outputShearX")
	outputShearY_ : OutputShearYPlug = PlugDescriptor("outputShearY")
	outputShearZ_ : OutputShearZPlug = PlugDescriptor("outputShearZ")
	outputShear_ : OutputShearPlug = PlugDescriptor("outputShear")
	outputTranslateX_ : OutputTranslateXPlug = PlugDescriptor("outputTranslateX")
	outputTranslateY_ : OutputTranslateYPlug = PlugDescriptor("outputTranslateY")
	outputTranslateZ_ : OutputTranslateZPlug = PlugDescriptor("outputTranslateZ")
	outputTranslate_ : OutputTranslatePlug = PlugDescriptor("outputTranslate")

	# node attributes

	typeName = "decomposeMatrix"
	apiTypeInt = 1135
	apiTypeStr = "kDecomposeMatrix"
	typeIdInt = 1476395776
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "inputMatrix", "inputRotateOrder", "outputQuatW", "outputQuatX", "outputQuatY", "outputQuatZ", "outputQuat", "outputRotateX", "outputRotateY", "outputRotateZ", "outputRotate", "outputScaleX", "outputScaleY", "outputScaleZ", "outputScale", "outputShearX", "outputShearY", "outputShearZ", "outputShear", "outputTranslateX", "outputTranslateY", "outputTranslateZ", "outputTranslate"]
	nodeLeafPlugs = ["binMembership", "inputMatrix", "inputRotateOrder", "outputQuat", "outputRotate", "outputScale", "outputShear", "outputTranslate"]
	pass

