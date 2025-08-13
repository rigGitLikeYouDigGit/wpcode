

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
	node : Luminance = None
	pass
class OutValuePlug(Plug):
	node : Luminance = None
	pass
class RenderPassModePlug(Plug):
	node : Luminance = None
	pass
class ValueBPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : Luminance = None
	pass
class ValueGPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : Luminance = None
	pass
class ValueRPlug(Plug):
	parent : ValuePlug = PlugDescriptor("value")
	node : Luminance = None
	pass
class ValuePlug(Plug):
	valueB_ : ValueBPlug = PlugDescriptor("valueB")
	vb_ : ValueBPlug = PlugDescriptor("valueB")
	valueG_ : ValueGPlug = PlugDescriptor("valueG")
	vg_ : ValueGPlug = PlugDescriptor("valueG")
	valueR_ : ValueRPlug = PlugDescriptor("valueR")
	vr_ : ValueRPlug = PlugDescriptor("valueR")
	node : Luminance = None
	pass
# endregion


# define node class
class Luminance(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	outValue_ : OutValuePlug = PlugDescriptor("outValue")
	renderPassMode_ : RenderPassModePlug = PlugDescriptor("renderPassMode")
	valueB_ : ValueBPlug = PlugDescriptor("valueB")
	valueG_ : ValueGPlug = PlugDescriptor("valueG")
	valueR_ : ValueRPlug = PlugDescriptor("valueR")
	value_ : ValuePlug = PlugDescriptor("value")

	# node attributes

	typeName = "luminance"
	apiTypeInt = 384
	apiTypeStr = "kLuminance"
	typeIdInt = 1380734285
	MFnCls = om.MFnDependencyNode
	pass

