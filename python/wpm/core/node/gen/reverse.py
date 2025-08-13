

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
	node : Reverse = None
	pass
class InputXPlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	node : Reverse = None
	pass
class InputYPlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	node : Reverse = None
	pass
class InputZPlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	node : Reverse = None
	pass
class InputPlug(Plug):
	inputX_ : InputXPlug = PlugDescriptor("inputX")
	ix_ : InputXPlug = PlugDescriptor("inputX")
	inputY_ : InputYPlug = PlugDescriptor("inputY")
	iy_ : InputYPlug = PlugDescriptor("inputY")
	inputZ_ : InputZPlug = PlugDescriptor("inputZ")
	iz_ : InputZPlug = PlugDescriptor("inputZ")
	node : Reverse = None
	pass
class OutputXPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : Reverse = None
	pass
class OutputYPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : Reverse = None
	pass
class OutputZPlug(Plug):
	parent : OutputPlug = PlugDescriptor("output")
	node : Reverse = None
	pass
class OutputPlug(Plug):
	outputX_ : OutputXPlug = PlugDescriptor("outputX")
	ox_ : OutputXPlug = PlugDescriptor("outputX")
	outputY_ : OutputYPlug = PlugDescriptor("outputY")
	oy_ : OutputYPlug = PlugDescriptor("outputY")
	outputZ_ : OutputZPlug = PlugDescriptor("outputZ")
	oz_ : OutputZPlug = PlugDescriptor("outputZ")
	node : Reverse = None
	pass
# endregion


# define node class
class Reverse(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	inputX_ : InputXPlug = PlugDescriptor("inputX")
	inputY_ : InputYPlug = PlugDescriptor("inputY")
	inputZ_ : InputZPlug = PlugDescriptor("inputZ")
	input_ : InputPlug = PlugDescriptor("input")
	outputX_ : OutputXPlug = PlugDescriptor("outputX")
	outputY_ : OutputYPlug = PlugDescriptor("outputY")
	outputZ_ : OutputZPlug = PlugDescriptor("outputZ")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "reverse"
	apiTypeInt = 468
	apiTypeStr = "kReverse"
	typeIdInt = 1381127763
	MFnCls = om.MFnDependencyNode
	pass

