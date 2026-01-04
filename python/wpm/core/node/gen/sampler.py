

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : Sampler = None
	pass
class Function_DefaultPlug(Plug):
	parent : FunctionPlug = PlugDescriptor("function")
	node : Sampler = None
	pass
class Function_HiddenPlug(Plug):
	parent : FunctionPlug = PlugDescriptor("function")
	node : Sampler = None
	pass
class Function_InmapFromPlug(Plug):
	parent : Function_InmapPlug = PlugDescriptor("function_Inmap")
	node : Sampler = None
	pass
class Function_InmapToPlug(Plug):
	parent : Function_InmapPlug = PlugDescriptor("function_Inmap")
	node : Sampler = None
	pass
class Function_InmapPlug(Plug):
	parent : FunctionPlug = PlugDescriptor("function")
	function_InmapFrom_ : Function_InmapFromPlug = PlugDescriptor("function_InmapFrom")
	fif_ : Function_InmapFromPlug = PlugDescriptor("function_InmapFrom")
	function_InmapTo_ : Function_InmapToPlug = PlugDescriptor("function_InmapTo")
	fit_ : Function_InmapToPlug = PlugDescriptor("function_InmapTo")
	node : Sampler = None
	pass
class Function_OutmapFromPlug(Plug):
	parent : Function_OutmapPlug = PlugDescriptor("function_Outmap")
	node : Sampler = None
	pass
class Function_OutmapToPlug(Plug):
	parent : Function_OutmapPlug = PlugDescriptor("function_Outmap")
	node : Sampler = None
	pass
class Function_OutmapPlug(Plug):
	parent : FunctionPlug = PlugDescriptor("function")
	function_OutmapFrom_ : Function_OutmapFromPlug = PlugDescriptor("function_OutmapFrom")
	fof_ : Function_OutmapFromPlug = PlugDescriptor("function_OutmapFrom")
	function_OutmapTo_ : Function_OutmapToPlug = PlugDescriptor("function_OutmapTo")
	fot_ : Function_OutmapToPlug = PlugDescriptor("function_OutmapTo")
	node : Sampler = None
	pass
class Function_RawPlug(Plug):
	parent : FunctionPlug = PlugDescriptor("function")
	node : Sampler = None
	pass
class FunctionPlug(Plug):
	function_Default_ : Function_DefaultPlug = PlugDescriptor("function_Default")
	fd_ : Function_DefaultPlug = PlugDescriptor("function_Default")
	function_Hidden_ : Function_HiddenPlug = PlugDescriptor("function_Hidden")
	fh_ : Function_HiddenPlug = PlugDescriptor("function_Hidden")
	function_Inmap_ : Function_InmapPlug = PlugDescriptor("function_Inmap")
	fi_ : Function_InmapPlug = PlugDescriptor("function_Inmap")
	function_Outmap_ : Function_OutmapPlug = PlugDescriptor("function_Outmap")
	fo_ : Function_OutmapPlug = PlugDescriptor("function_Outmap")
	function_Raw_ : Function_RawPlug = PlugDescriptor("function_Raw")
	fr_ : Function_RawPlug = PlugDescriptor("function_Raw")
	node : Sampler = None
	pass
class InvertPlug(Plug):
	node : Sampler = None
	pass
class MaximumPlug(Plug):
	node : Sampler = None
	pass
class MinimumPlug(Plug):
	node : Sampler = None
	pass
class StepPlug(Plug):
	node : Sampler = None
	pass
class ValuePlug(Plug):
	node : Sampler = None
	pass
# endregion


# define node class
class Sampler(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	function_Default_ : Function_DefaultPlug = PlugDescriptor("function_Default")
	function_Hidden_ : Function_HiddenPlug = PlugDescriptor("function_Hidden")
	function_InmapFrom_ : Function_InmapFromPlug = PlugDescriptor("function_InmapFrom")
	function_InmapTo_ : Function_InmapToPlug = PlugDescriptor("function_InmapTo")
	function_Inmap_ : Function_InmapPlug = PlugDescriptor("function_Inmap")
	function_OutmapFrom_ : Function_OutmapFromPlug = PlugDescriptor("function_OutmapFrom")
	function_OutmapTo_ : Function_OutmapToPlug = PlugDescriptor("function_OutmapTo")
	function_Outmap_ : Function_OutmapPlug = PlugDescriptor("function_Outmap")
	function_Raw_ : Function_RawPlug = PlugDescriptor("function_Raw")
	function_ : FunctionPlug = PlugDescriptor("function")
	invert_ : InvertPlug = PlugDescriptor("invert")
	maximum_ : MaximumPlug = PlugDescriptor("maximum")
	minimum_ : MinimumPlug = PlugDescriptor("minimum")
	step_ : StepPlug = PlugDescriptor("step")
	value_ : ValuePlug = PlugDescriptor("value")

	# node attributes

	typeName = "sampler"
	typeIdInt = 1179864400
	nodeLeafClassAttrs = ["binMembership", "function_Default", "function_Hidden", "function_InmapFrom", "function_InmapTo", "function_Inmap", "function_OutmapFrom", "function_OutmapTo", "function_Outmap", "function_Raw", "function", "invert", "maximum", "minimum", "step", "value"]
	nodeLeafPlugs = ["binMembership", "function", "invert", "maximum", "minimum", "step", "value"]
	pass

