

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
	node : TimeToUnitConversion = None
	pass
class ConversionFactorPlug(Plug):
	node : TimeToUnitConversion = None
	pass
class InputPlug(Plug):
	node : TimeToUnitConversion = None
	pass
class OutputPlug(Plug):
	node : TimeToUnitConversion = None
	pass
# endregion


# define node class
class TimeToUnitConversion(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	conversionFactor_ : ConversionFactorPlug = PlugDescriptor("conversionFactor")
	input_ : InputPlug = PlugDescriptor("input")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "timeToUnitConversion"
	apiTypeInt = 521
	apiTypeStr = "kTimeToUnitConversion"
	typeIdInt = 1146375509
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "conversionFactor", "input", "output"]
	nodeLeafPlugs = ["binMembership", "conversionFactor", "input", "output"]
	pass

