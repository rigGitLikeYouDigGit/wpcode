

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
	node : MatrixCurve = None
	pass
class CachingPlug(Plug):
	node : MatrixCurve = None
	pass
class CurveOutPlug(Plug):
	node : MatrixCurve = None
	pass
class CurveRootResInPlug(Plug):
	node : MatrixCurve = None
	pass
class FrozenPlug(Plug):
	node : MatrixCurve = None
	pass
class IsHistoricallyInterestingPlug(Plug):
	node : MatrixCurve = None
	pass
class MatrixEndInPlug(Plug):
	node : MatrixCurve = None
	pass
class MatrixMidInMatrixPlug(Plug):
	parent : MatrixMidInPlug = PlugDescriptor("matrixMidIn")
	node : MatrixCurve = None
	pass
class MatrixMidInPlug(Plug):
	matrixMidInMatrix_ : MatrixMidInMatrixPlug = PlugDescriptor("matrixMidInMatrix")
	matrixMidInMatrix_ : MatrixMidInMatrixPlug = PlugDescriptor("matrixMidInMatrix")
	node : MatrixCurve = None
	pass
class MatrixRootIterationsInPlug(Plug):
	node : MatrixCurve = None
	pass
class MatrixStartInPlug(Plug):
	node : MatrixCurve = None
	pass
class MessagePlug(Plug):
	node : MatrixCurve = None
	pass
class NodeStatePlug(Plug):
	node : MatrixCurve = None
	pass
class SampleInParamPlug(Plug):
	parent : SampleInPlug = PlugDescriptor("sampleIn")
	node : MatrixCurve = None
	pass
class SampleInPlug(Plug):
	sampleInParam_ : SampleInParamPlug = PlugDescriptor("sampleInParam")
	sampleInParam_ : SampleInParamPlug = PlugDescriptor("sampleInParam")
	node : MatrixCurve = None
	pass
class SampleOutMatrixPlug(Plug):
	parent : SampleOutPlug = PlugDescriptor("sampleOut")
	node : MatrixCurve = None
	pass
class SampleOutPlug(Plug):
	sampleOutMatrix_ : SampleOutMatrixPlug = PlugDescriptor("sampleOutMatrix")
	sampleOutMatrix_ : SampleOutMatrixPlug = PlugDescriptor("sampleOutMatrix")
	node : MatrixCurve = None
	pass
# endregion


# define node class
class MatrixCurve(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	caching_ : CachingPlug = PlugDescriptor("caching")
	curveOut_ : CurveOutPlug = PlugDescriptor("curveOut")
	curveRootResIn_ : CurveRootResInPlug = PlugDescriptor("curveRootResIn")
	frozen_ : FrozenPlug = PlugDescriptor("frozen")
	isHistoricallyInteresting_ : IsHistoricallyInterestingPlug = PlugDescriptor("isHistoricallyInteresting")
	matrixEndIn_ : MatrixEndInPlug = PlugDescriptor("matrixEndIn")
	matrixMidInMatrix_ : MatrixMidInMatrixPlug = PlugDescriptor("matrixMidInMatrix")
	matrixMidIn_ : MatrixMidInPlug = PlugDescriptor("matrixMidIn")
	matrixRootIterationsIn_ : MatrixRootIterationsInPlug = PlugDescriptor("matrixRootIterationsIn")
	matrixStartIn_ : MatrixStartInPlug = PlugDescriptor("matrixStartIn")
	message_ : MessagePlug = PlugDescriptor("message")
	nodeState_ : NodeStatePlug = PlugDescriptor("nodeState")
	sampleInParam_ : SampleInParamPlug = PlugDescriptor("sampleInParam")
	sampleIn_ : SampleInPlug = PlugDescriptor("sampleIn")
	sampleOutMatrix_ : SampleOutMatrixPlug = PlugDescriptor("sampleOutMatrix")
	sampleOut_ : SampleOutPlug = PlugDescriptor("sampleOut")

	# node attributes

	typeName = "matrixCurve"
	typeIdInt = 1191076
	pass

