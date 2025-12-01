

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
	node : Clamp = None
	pass
class InputBPlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	node : Clamp = None
	pass
class InputGPlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	node : Clamp = None
	pass
class InputRPlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	node : Clamp = None
	pass
class InputPlug(Plug):
	inputB_ : InputBPlug = PlugDescriptor("inputB")
	ipb_ : InputBPlug = PlugDescriptor("inputB")
	inputG_ : InputGPlug = PlugDescriptor("inputG")
	ipg_ : InputGPlug = PlugDescriptor("inputG")
	inputR_ : InputRPlug = PlugDescriptor("inputR")
	ipr_ : InputRPlug = PlugDescriptor("inputR")
	node : Clamp = None
	pass
class MaxBPlug(Plug):
	parent : MaxPlug = PlugDescriptor("max")
	node : Clamp = None
	pass
class MaxGPlug(Plug):
	parent : MaxPlug = PlugDescriptor("max")
	node : Clamp = None
	pass
class MaxRPlug(Plug):
	parent : MaxPlug = PlugDescriptor("max")
	node : Clamp = None
	pass
class MaxPlug(Plug):
	maxB_ : MaxBPlug = PlugDescriptor("maxB")
	mxb_ : MaxBPlug = PlugDescriptor("maxB")
	maxG_ : MaxGPlug = PlugDescriptor("maxG")
	mxg_ : MaxGPlug = PlugDescriptor("maxG")
	maxR_ : MaxRPlug = PlugDescriptor("maxR")
	mxr_ : MaxRPlug = PlugDescriptor("maxR")
	node : Clamp = None
	pass
class MinBPlug(Plug):
	parent : MinPlug = PlugDescriptor("min")
	node : Clamp = None
	pass
class MinGPlug(Plug):
	parent : MinPlug = PlugDescriptor("min")
	node : Clamp = None
	pass
class MinRPlug(Plug):
	parent : MinPlug = PlugDescriptor("min")
	node : Clamp = None
	pass
class MinPlug(Plug):
	minB_ : MinBPlug = PlugDescriptor("minB")
	mnb_ : MinBPlug = PlugDescriptor("minB")
	minG_ : MinGPlug = PlugDescriptor("minG")
	mng_ : MinGPlug = PlugDescriptor("minG")
	minR_ : MinRPlug = PlugDescriptor("minR")
	mnr_ : MinRPlug = PlugDescriptor("minR")
	node : Clamp = None
	pass
class OutputBPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : Clamp = None
	pass
class OutputGPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : Clamp = None
	pass
class OutputRPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : Clamp = None
	pass
class OutputPlug(Plug):
	outputB_ : OutputBPlug = PlugDescriptor("outputB")
	opb_ : OutputBPlug = PlugDescriptor("outputB")
	outputG_ : OutputGPlug = PlugDescriptor("outputG")
	opg_ : OutputGPlug = PlugDescriptor("outputG")
	outputR_ : OutputRPlug = PlugDescriptor("outputR")
	opr_ : OutputRPlug = PlugDescriptor("outputR")
	node : Clamp = None
	pass
class RenderPassModePlug(Plug):
	node : Clamp = None
	pass
# endregion


# define node class
class Clamp(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	inputB_ : InputBPlug = PlugDescriptor("inputB")
	inputG_ : InputGPlug = PlugDescriptor("inputG")
	inputR_ : InputRPlug = PlugDescriptor("inputR")
	input_ : InputPlug = PlugDescriptor("input")
	maxB_ : MaxBPlug = PlugDescriptor("maxB")
	maxG_ : MaxGPlug = PlugDescriptor("maxG")
	maxR_ : MaxRPlug = PlugDescriptor("maxR")
	max_ : MaxPlug = PlugDescriptor("max")
	minB_ : MinBPlug = PlugDescriptor("minB")
	minG_ : MinGPlug = PlugDescriptor("minG")
	minR_ : MinRPlug = PlugDescriptor("minR")
	min_ : MinPlug = PlugDescriptor("min")
	outputB_ : OutputBPlug = PlugDescriptor("outputB")
	outputG_ : OutputGPlug = PlugDescriptor("outputG")
	outputR_ : OutputRPlug = PlugDescriptor("outputR")
	output_ : OutputPlug = PlugDescriptor("output")
	renderPassMode_ : RenderPassModePlug = PlugDescriptor("renderPassMode")

	# node attributes

	typeName = "clamp"
	typeIdInt = 1380142131
	nodeLeafClassAttrs = ["binMembership", "inputB", "inputG", "inputR", "input", "maxB", "maxG", "maxR", "max", "minB", "minG", "minR", "min", "outputB", "outputG", "outputR", "output", "renderPassMode"]
	nodeLeafPlugs = ["binMembership", "input", "max", "min", "output", "renderPassMode"]
	pass

