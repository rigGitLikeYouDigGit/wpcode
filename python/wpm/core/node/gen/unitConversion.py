

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
	node : UnitConversion = None
	pass
class ConversionFactorPlug(Plug):
	node : UnitConversion = None
	pass
class InputPlug(Plug):
	node : UnitConversion = None
	pass
class OutputPlug(Plug):
	node : UnitConversion = None
	pass
# endregion


# define node class
class UnitConversion(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	conversionFactor_ : ConversionFactorPlug = PlugDescriptor("conversionFactor")
	input_ : InputPlug = PlugDescriptor("input")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "unitConversion"
	apiTypeInt = 529
	apiTypeStr = "kUnitConversion"
	typeIdInt = 1146441300
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "conversionFactor", "input", "output"]
	nodeLeafPlugs = ["binMembership", "conversionFactor", "input", "output"]
	pass

