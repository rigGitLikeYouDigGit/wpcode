

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
	node : CombinationShape = None
	pass
class CombinationMethodPlug(Plug):
	node : CombinationShape = None
	pass
class InputWeightPlug(Plug):
	node : CombinationShape = None
	pass
class OutputWeightPlug(Plug):
	node : CombinationShape = None
	pass
# endregion


# define node class
class CombinationShape(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	combinationMethod_ : CombinationMethodPlug = PlugDescriptor("combinationMethod")
	inputWeight_ : InputWeightPlug = PlugDescriptor("inputWeight")
	outputWeight_ : OutputWeightPlug = PlugDescriptor("outputWeight")

	# node attributes

	typeName = "combinationShape"
	apiTypeInt = 337
	apiTypeStr = "kCombinationShape"
	typeIdInt = 1178816083
	MFnCls = om.MFnGeometryFilter
	nodeLeafClassAttrs = ["binMembership", "combinationMethod", "inputWeight", "outputWeight"]
	nodeLeafPlugs = ["binMembership", "combinationMethod", "inputWeight", "outputWeight"]
	pass

