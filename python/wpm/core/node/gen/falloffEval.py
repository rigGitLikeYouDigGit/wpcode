

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
	node : FalloffEval = None
	pass
class ComponentTagExpressionPlug(Plug):
	node : FalloffEval = None
	pass
class CurrentGeometryPlug(Plug):
	node : FalloffEval = None
	pass
class OriginalGeometryPlug(Plug):
	node : FalloffEval = None
	pass
class PerFunctionVertexWeightsPlug(Plug):
	parent : PerFunctionWeightsPlug = PlugDescriptor("perFunctionWeights")
	node : FalloffEval = None
	pass
class PerFunctionWeightsPlug(Plug):
	perFunctionVertexWeights_ : PerFunctionVertexWeightsPlug = PlugDescriptor("perFunctionVertexWeights")
	pfvw_ : PerFunctionVertexWeightsPlug = PlugDescriptor("perFunctionVertexWeights")
	node : FalloffEval = None
	pass
class PerVertexFalloffWeightsPlug(Plug):
	parent : PerVertexWeightsPlug = PlugDescriptor("perVertexWeights")
	node : FalloffEval = None
	pass
class PerVertexWeightsPlug(Plug):
	perVertexFalloffWeights_ : PerVertexFalloffWeightsPlug = PlugDescriptor("perVertexFalloffWeights")
	pvfw_ : PerVertexFalloffWeightsPlug = PlugDescriptor("perVertexFalloffWeights")
	node : FalloffEval = None
	pass
class WeightFunctionPlug(Plug):
	node : FalloffEval = None
	pass
# endregion


# define node class
class FalloffEval(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	componentTagExpression_ : ComponentTagExpressionPlug = PlugDescriptor("componentTagExpression")
	currentGeometry_ : CurrentGeometryPlug = PlugDescriptor("currentGeometry")
	originalGeometry_ : OriginalGeometryPlug = PlugDescriptor("originalGeometry")
	perFunctionVertexWeights_ : PerFunctionVertexWeightsPlug = PlugDescriptor("perFunctionVertexWeights")
	perFunctionWeights_ : PerFunctionWeightsPlug = PlugDescriptor("perFunctionWeights")
	perVertexFalloffWeights_ : PerVertexFalloffWeightsPlug = PlugDescriptor("perVertexFalloffWeights")
	perVertexWeights_ : PerVertexWeightsPlug = PlugDescriptor("perVertexWeights")
	weightFunction_ : WeightFunctionPlug = PlugDescriptor("weightFunction")

	# node attributes

	typeName = "falloffEval"
	apiTypeInt = 1148
	apiTypeStr = "kFalloffEval"
	typeIdInt = 1162235718
	MFnCls = om.MFnDependencyNode
	pass

