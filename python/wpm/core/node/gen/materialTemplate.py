

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
class ComponentTagExpressionPlug(Plug):
	parent : AssignPlug = PlugDescriptor("assign")
	node : MaterialTemplate = None
	pass
class ShadingEnginePlug(Plug):
	parent : AssignPlug = PlugDescriptor("assign")
	node : MaterialTemplate = None
	pass
class AssignPlug(Plug):
	componentTagExpression_ : ComponentTagExpressionPlug = PlugDescriptor("componentTagExpression")
	gtg_ : ComponentTagExpressionPlug = PlugDescriptor("componentTagExpression")
	shadingEngine_ : ShadingEnginePlug = PlugDescriptor("shadingEngine")
	shd_ : ShadingEnginePlug = PlugDescriptor("shadingEngine")
	node : MaterialTemplate = None
	pass
class BinMembershipPlug(Plug):
	node : MaterialTemplate = None
	pass
class DefaultShadingEnginePlug(Plug):
	node : MaterialTemplate = None
	pass
# endregion


# define node class
class MaterialTemplate(_BASE_):
	componentTagExpression_ : ComponentTagExpressionPlug = PlugDescriptor("componentTagExpression")
	shadingEngine_ : ShadingEnginePlug = PlugDescriptor("shadingEngine")
	assign_ : AssignPlug = PlugDescriptor("assign")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	defaultShadingEngine_ : DefaultShadingEnginePlug = PlugDescriptor("defaultShadingEngine")

	# node attributes

	typeName = "materialTemplate"
	apiTypeInt = 393
	apiTypeStr = "kMaterialTemplate"
	typeIdInt = 1297370177
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["componentTagExpression", "shadingEngine", "assign", "binMembership", "defaultShadingEngine"]
	nodeLeafPlugs = ["assign", "binMembership", "defaultShadingEngine"]
	pass

