

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
	node : UnitToTimeConversion = None
	pass
class ConversionFactorPlug(Plug):
	node : UnitToTimeConversion = None
	pass
class InputPlug(Plug):
	node : UnitToTimeConversion = None
	pass
class OutputPlug(Plug):
	node : UnitToTimeConversion = None
	pass
# endregion


# define node class
class UnitToTimeConversion(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	conversionFactor_ : ConversionFactorPlug = PlugDescriptor("conversionFactor")
	input_ : InputPlug = PlugDescriptor("input")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "unitToTimeConversion"
	apiTypeInt = 530
	apiTypeStr = "kUnitToTimeConversion"
	typeIdInt = 1146442829
	MFnCls = om.MFnDependencyNode
	pass

