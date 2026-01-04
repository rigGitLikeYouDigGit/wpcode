

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
	node : MatrixCurve = None
	pass
class CurveOutPlug(Plug):
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
class MatrixStartInPlug(Plug):
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
	curveOut_ : CurveOutPlug = PlugDescriptor("curveOut")
	matrixEndIn_ : MatrixEndInPlug = PlugDescriptor("matrixEndIn")
	matrixMidInMatrix_ : MatrixMidInMatrixPlug = PlugDescriptor("matrixMidInMatrix")
	matrixMidIn_ : MatrixMidInPlug = PlugDescriptor("matrixMidIn")
	matrixStartIn_ : MatrixStartInPlug = PlugDescriptor("matrixStartIn")
	sampleInParam_ : SampleInParamPlug = PlugDescriptor("sampleInParam")
	sampleIn_ : SampleInPlug = PlugDescriptor("sampleIn")
	sampleOutMatrix_ : SampleOutMatrixPlug = PlugDescriptor("sampleOutMatrix")
	sampleOut_ : SampleOutPlug = PlugDescriptor("sampleOut")

	# node attributes

	typeName = "matrixCurve"
	typeIdInt = 1191076
	nodeLeafClassAttrs = ["binMembership", "curveOut", "matrixEndIn", "matrixMidInMatrix", "matrixMidIn", "matrixStartIn", "sampleInParam", "sampleIn", "sampleOutMatrix", "sampleOut"]
	nodeLeafPlugs = ["binMembership", "curveOut", "matrixEndIn", "matrixMidIn", "matrixStartIn", "sampleIn", "sampleOut"]
	pass

