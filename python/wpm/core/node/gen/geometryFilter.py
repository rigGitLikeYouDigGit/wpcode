

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
	node : GeometryFilter = None
	pass
class BlockGPUPlug(Plug):
	node : GeometryFilter = None
	pass
class EnvelopePlug(Plug):
	node : GeometryFilter = None
	pass
class EnvelopeWeightsPlug(Plug):
	parent : EnvelopeWeightsListPlug = PlugDescriptor("envelopeWeightsList")
	node : GeometryFilter = None
	pass
class EnvelopeWeightsListPlug(Plug):
	envelopeWeights_ : EnvelopeWeightsPlug = PlugDescriptor("envelopeWeights")
	owt_ : EnvelopeWeightsPlug = PlugDescriptor("envelopeWeights")
	node : GeometryFilter = None
	pass
class Fchild1Plug(Plug):
	parent : FunctionPlug = PlugDescriptor("function")
	node : GeometryFilter = None
	pass
class Fchild2Plug(Plug):
	parent : FunctionPlug = PlugDescriptor("function")
	node : GeometryFilter = None
	pass
class Fchild3Plug(Plug):
	parent : FunctionPlug = PlugDescriptor("function")
	node : GeometryFilter = None
	pass
class FunctionPlug(Plug):
	fchild1_ : Fchild1Plug = PlugDescriptor("fchild1")
	f1_ : Fchild1Plug = PlugDescriptor("fchild1")
	fchild2_ : Fchild2Plug = PlugDescriptor("fchild2")
	f2_ : Fchild2Plug = PlugDescriptor("fchild2")
	fchild3_ : Fchild3Plug = PlugDescriptor("fchild3")
	f3_ : Fchild3Plug = PlugDescriptor("fchild3")
	node : GeometryFilter = None
	pass
class ComponentTagExpressionPlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	node : GeometryFilter = None
	pass
class GroupIdPlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	node : GeometryFilter = None
	pass
class InputGeometryPlug(Plug):
	parent : InputPlug = PlugDescriptor("input")
	node : GeometryFilter = None
	pass
class InputPlug(Plug):
	componentTagExpression_ : ComponentTagExpressionPlug = PlugDescriptor("componentTagExpression")
	gtg_ : ComponentTagExpressionPlug = PlugDescriptor("componentTagExpression")
	groupId_ : GroupIdPlug = PlugDescriptor("groupId")
	gi_ : GroupIdPlug = PlugDescriptor("groupId")
	inputGeometry_ : InputGeometryPlug = PlugDescriptor("inputGeometry")
	ig_ : InputGeometryPlug = PlugDescriptor("inputGeometry")
	node : GeometryFilter = None
	pass
class Map64BitIndicesPlug(Plug):
	node : GeometryFilter = None
	pass
class OriginalGeometryPlug(Plug):
	node : GeometryFilter = None
	pass
class OutputGeometryPlug(Plug):
	node : GeometryFilter = None
	pass
class WeightFunctionPlug(Plug):
	node : GeometryFilter = None
	pass
# endregion


# define node class
class GeometryFilter(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	blockGPU_ : BlockGPUPlug = PlugDescriptor("blockGPU")
	envelope_ : EnvelopePlug = PlugDescriptor("envelope")
	envelopeWeights_ : EnvelopeWeightsPlug = PlugDescriptor("envelopeWeights")
	envelopeWeightsList_ : EnvelopeWeightsListPlug = PlugDescriptor("envelopeWeightsList")
	fchild1_ : Fchild1Plug = PlugDescriptor("fchild1")
	fchild2_ : Fchild2Plug = PlugDescriptor("fchild2")
	fchild3_ : Fchild3Plug = PlugDescriptor("fchild3")
	function_ : FunctionPlug = PlugDescriptor("function")
	componentTagExpression_ : ComponentTagExpressionPlug = PlugDescriptor("componentTagExpression")
	groupId_ : GroupIdPlug = PlugDescriptor("groupId")
	inputGeometry_ : InputGeometryPlug = PlugDescriptor("inputGeometry")
	input_ : InputPlug = PlugDescriptor("input")
	map64BitIndices_ : Map64BitIndicesPlug = PlugDescriptor("map64BitIndices")
	originalGeometry_ : OriginalGeometryPlug = PlugDescriptor("originalGeometry")
	outputGeometry_ : OutputGeometryPlug = PlugDescriptor("outputGeometry")
	weightFunction_ : WeightFunctionPlug = PlugDescriptor("weightFunction")

	# node attributes

	typeName = "geometryFilter"
	typeIdInt = 1145521737
	nodeLeafClassAttrs = ["binMembership", "blockGPU", "envelope", "envelopeWeights", "envelopeWeightsList", "fchild1", "fchild2", "fchild3", "function", "componentTagExpression", "groupId", "inputGeometry", "input", "map64BitIndices", "originalGeometry", "outputGeometry", "weightFunction"]
	nodeLeafPlugs = ["binMembership", "blockGPU", "envelope", "envelopeWeightsList", "function", "input", "map64BitIndices", "originalGeometry", "outputGeometry", "weightFunction"]
	pass

