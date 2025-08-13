

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
	node : ComponentMatch = None
	pass
class ComponentLookupPlug(Plug):
	node : ComponentMatch = None
	pass
class ComponentTagExpressionPlug(Plug):
	node : ComponentMatch = None
	pass
class InputGeometryPlug(Plug):
	node : ComponentMatch = None
	pass
class MatchModePlug(Plug):
	node : ComponentMatch = None
	pass
class TargetGeometryPlug(Plug):
	node : ComponentMatch = None
	pass
class UniqueMatchPlug(Plug):
	node : ComponentMatch = None
	pass
# endregion


# define node class
class ComponentMatch(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	componentLookup_ : ComponentLookupPlug = PlugDescriptor("componentLookup")
	componentTagExpression_ : ComponentTagExpressionPlug = PlugDescriptor("componentTagExpression")
	inputGeometry_ : InputGeometryPlug = PlugDescriptor("inputGeometry")
	matchMode_ : MatchModePlug = PlugDescriptor("matchMode")
	targetGeometry_ : TargetGeometryPlug = PlugDescriptor("targetGeometry")
	uniqueMatch_ : UniqueMatchPlug = PlugDescriptor("uniqueMatch")

	# node attributes

	typeName = "componentMatch"
	apiTypeInt = 1149
	apiTypeStr = "kComponentMatch"
	typeIdInt = 1414484045
	MFnCls = om.MFnDependencyNode
	pass

