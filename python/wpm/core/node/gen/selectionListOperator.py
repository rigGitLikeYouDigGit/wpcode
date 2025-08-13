

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
	node : SelectionListOperator = None
	pass
class InputListAPlug(Plug):
	node : SelectionListOperator = None
	pass
class InputListBPlug(Plug):
	node : SelectionListOperator = None
	pass
class OperationPlug(Plug):
	node : SelectionListOperator = None
	pass
class OperatorClassPlug(Plug):
	node : SelectionListOperator = None
	pass
class OutputListPlug(Plug):
	node : SelectionListOperator = None
	pass
# endregion


# define node class
class SelectionListOperator(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	inputListA_ : InputListAPlug = PlugDescriptor("inputListA")
	inputListB_ : InputListBPlug = PlugDescriptor("inputListB")
	operation_ : OperationPlug = PlugDescriptor("operation")
	operatorClass_ : OperatorClassPlug = PlugDescriptor("operatorClass")
	outputList_ : OutputListPlug = PlugDescriptor("outputList")

	# node attributes

	typeName = "selectionListOperator"
	apiTypeInt = 683
	apiTypeStr = "kSelectionListOperator"
	typeIdInt = 1397509968
	MFnCls = om.MFnDependencyNode
	pass

