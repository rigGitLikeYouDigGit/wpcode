

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AnimBlendNodeBase = retriever.getNodeCls("AnimBlendNodeBase")
assert AnimBlendNodeBase
if T.TYPE_CHECKING:
	from .. import AnimBlendNodeBase

# add node doc



# region plug type defs
class AccumulationModePlug(Plug):
	node : AnimBlendNodeAdditiveRotation = None
	pass
class ByLayerAccLegacyModePlug(Plug):
	node : AnimBlendNodeAdditiveRotation = None
	pass
class InputAXPlug(Plug):
	parent : InputAPlug = PlugDescriptor("inputA")
	node : AnimBlendNodeAdditiveRotation = None
	pass
class InputAYPlug(Plug):
	parent : InputAPlug = PlugDescriptor("inputA")
	node : AnimBlendNodeAdditiveRotation = None
	pass
class InputAZPlug(Plug):
	parent : InputAPlug = PlugDescriptor("inputA")
	node : AnimBlendNodeAdditiveRotation = None
	pass
class InputAPlug(Plug):
	inputAX_ : InputAXPlug = PlugDescriptor("inputAX")
	iax_ : InputAXPlug = PlugDescriptor("inputAX")
	inputAY_ : InputAYPlug = PlugDescriptor("inputAY")
	iay_ : InputAYPlug = PlugDescriptor("inputAY")
	inputAZ_ : InputAZPlug = PlugDescriptor("inputAZ")
	iaz_ : InputAZPlug = PlugDescriptor("inputAZ")
	node : AnimBlendNodeAdditiveRotation = None
	pass
class InputBXPlug(Plug):
	parent : InputBPlug = PlugDescriptor("inputB")
	node : AnimBlendNodeAdditiveRotation = None
	pass
class InputBYPlug(Plug):
	parent : InputBPlug = PlugDescriptor("inputB")
	node : AnimBlendNodeAdditiveRotation = None
	pass
class InputBZPlug(Plug):
	parent : InputBPlug = PlugDescriptor("inputB")
	node : AnimBlendNodeAdditiveRotation = None
	pass
class InputBPlug(Plug):
	inputBX_ : InputBXPlug = PlugDescriptor("inputBX")
	ibx_ : InputBXPlug = PlugDescriptor("inputBX")
	inputBY_ : InputBYPlug = PlugDescriptor("inputBY")
	iby_ : InputBYPlug = PlugDescriptor("inputBY")
	inputBZ_ : InputBZPlug = PlugDescriptor("inputBZ")
	ibz_ : InputBZPlug = PlugDescriptor("inputBZ")
	node : AnimBlendNodeAdditiveRotation = None
	pass
class OutputXPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : AnimBlendNodeAdditiveRotation = None
	pass
class OutputYPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : AnimBlendNodeAdditiveRotation = None
	pass
class OutputZPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : AnimBlendNodeAdditiveRotation = None
	pass
class OutputPlug(Plug):
	outputX_ : OutputXPlug = PlugDescriptor("outputX")
	ox_ : OutputXPlug = PlugDescriptor("outputX")
	outputY_ : OutputYPlug = PlugDescriptor("outputY")
	oy_ : OutputYPlug = PlugDescriptor("outputY")
	outputZ_ : OutputZPlug = PlugDescriptor("outputZ")
	oz_ : OutputZPlug = PlugDescriptor("outputZ")
	node : AnimBlendNodeAdditiveRotation = None
	pass
class RotateOrderPlug(Plug):
	node : AnimBlendNodeAdditiveRotation = None
	pass
class RotationInterpolationPlug(Plug):
	node : AnimBlendNodeAdditiveRotation = None
	pass
# endregion


# define node class
class AnimBlendNodeAdditiveRotation(AnimBlendNodeBase):
	accumulationMode_ : AccumulationModePlug = PlugDescriptor("accumulationMode")
	byLayerAccLegacyMode_ : ByLayerAccLegacyModePlug = PlugDescriptor("byLayerAccLegacyMode")
	inputAX_ : InputAXPlug = PlugDescriptor("inputAX")
	inputAY_ : InputAYPlug = PlugDescriptor("inputAY")
	inputAZ_ : InputAZPlug = PlugDescriptor("inputAZ")
	inputA_ : InputAPlug = PlugDescriptor("inputA")
	inputBX_ : InputBXPlug = PlugDescriptor("inputBX")
	inputBY_ : InputBYPlug = PlugDescriptor("inputBY")
	inputBZ_ : InputBZPlug = PlugDescriptor("inputBZ")
	inputB_ : InputBPlug = PlugDescriptor("inputB")
	outputX_ : OutputXPlug = PlugDescriptor("outputX")
	outputY_ : OutputYPlug = PlugDescriptor("outputY")
	outputZ_ : OutputZPlug = PlugDescriptor("outputZ")
	output_ : OutputPlug = PlugDescriptor("output")
	rotateOrder_ : RotateOrderPlug = PlugDescriptor("rotateOrder")
	rotationInterpolation_ : RotationInterpolationPlug = PlugDescriptor("rotationInterpolation")

	# node attributes

	typeName = "animBlendNodeAdditiveRotation"
	typeIdInt = 1094864466
	pass

