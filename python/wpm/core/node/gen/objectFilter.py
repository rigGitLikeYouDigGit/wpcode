

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
class AnnotationPlug(Plug):
	node : ObjectFilter = None
	pass
class BinMembershipPlug(Plug):
	node : ObjectFilter = None
	pass
class CategoryPlug(Plug):
	node : ObjectFilter = None
	pass
class ChildPlug(Plug):
	node : ObjectFilter = None
	pass
class DisablePlug(Plug):
	node : ObjectFilter = None
	pass
class FilterClassPlug(Plug):
	node : ObjectFilter = None
	pass
class InputListPlug(Plug):
	node : ObjectFilter = None
	pass
class InvertPlug(Plug):
	node : ObjectFilter = None
	pass
class OutputListPlug(Plug):
	node : ObjectFilter = None
	pass
# endregion


# define node class
class ObjectFilter(_BASE_):
	annotation_ : AnnotationPlug = PlugDescriptor("annotation")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	category_ : CategoryPlug = PlugDescriptor("category")
	child_ : ChildPlug = PlugDescriptor("child")
	disable_ : DisablePlug = PlugDescriptor("disable")
	filterClass_ : FilterClassPlug = PlugDescriptor("filterClass")
	inputList_ : InputListPlug = PlugDescriptor("inputList")
	invert_ : InvertPlug = PlugDescriptor("invert")
	outputList_ : OutputListPlug = PlugDescriptor("outputList")

	# node attributes

	typeName = "objectFilter"
	apiTypeInt = 676
	apiTypeStr = "kObjectFilter"
	typeIdInt = 1330007124
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["annotation", "binMembership", "category", "child", "disable", "filterClass", "inputList", "invert", "outputList"]
	nodeLeafPlugs = ["annotation", "binMembership", "category", "child", "disable", "filterClass", "inputList", "invert", "outputList"]
	pass

