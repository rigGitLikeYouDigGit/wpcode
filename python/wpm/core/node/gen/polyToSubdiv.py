

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
class AbsolutePositionPlug(Plug):
	node : PolyToSubdiv = None
	pass
class ApplyMatrixToResultPlug(Plug):
	node : PolyToSubdiv = None
	pass
class BinMembershipPlug(Plug):
	node : PolyToSubdiv = None
	pass
class UvPointsUPlug(Plug):
	parent : UvPointsPlug = PlugDescriptor("uvPoints")
	node : PolyToSubdiv = None
	pass
class UvPointsVPlug(Plug):
	parent : UvPointsPlug = PlugDescriptor("uvPoints")
	node : PolyToSubdiv = None
	pass
class UvPointsPlug(Plug):
	parent : CachedUVsPlug = PlugDescriptor("cachedUVs")
	uvPointsU_ : UvPointsUPlug = PlugDescriptor("uvPointsU")
	uvu_ : UvPointsUPlug = PlugDescriptor("uvPointsU")
	uvPointsV_ : UvPointsVPlug = PlugDescriptor("uvPointsV")
	uvv_ : UvPointsVPlug = PlugDescriptor("uvPointsV")
	node : PolyToSubdiv = None
	pass
class CachedUVsPlug(Plug):
	uvPoints_ : UvPointsPlug = PlugDescriptor("uvPoints")
	uvp_ : UvPointsPlug = PlugDescriptor("uvPoints")
	node : PolyToSubdiv = None
	pass
class InMeshPlug(Plug):
	node : PolyToSubdiv = None
	pass
class MaxEdgesPerVertPlug(Plug):
	node : PolyToSubdiv = None
	pass
class MaxPolyCountPlug(Plug):
	node : PolyToSubdiv = None
	pass
class OutSubdivPlug(Plug):
	node : PolyToSubdiv = None
	pass
class PreserveVertexOrderingPlug(Plug):
	node : PolyToSubdiv = None
	pass
class QuickConvertPlug(Plug):
	node : PolyToSubdiv = None
	pass
class UvTreatmentPlug(Plug):
	node : PolyToSubdiv = None
	pass
# endregion


# define node class
class PolyToSubdiv(_BASE_):
	absolutePosition_ : AbsolutePositionPlug = PlugDescriptor("absolutePosition")
	applyMatrixToResult_ : ApplyMatrixToResultPlug = PlugDescriptor("applyMatrixToResult")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	uvPointsU_ : UvPointsUPlug = PlugDescriptor("uvPointsU")
	uvPointsV_ : UvPointsVPlug = PlugDescriptor("uvPointsV")
	uvPoints_ : UvPointsPlug = PlugDescriptor("uvPoints")
	cachedUVs_ : CachedUVsPlug = PlugDescriptor("cachedUVs")
	inMesh_ : InMeshPlug = PlugDescriptor("inMesh")
	maxEdgesPerVert_ : MaxEdgesPerVertPlug = PlugDescriptor("maxEdgesPerVert")
	maxPolyCount_ : MaxPolyCountPlug = PlugDescriptor("maxPolyCount")
	outSubdiv_ : OutSubdivPlug = PlugDescriptor("outSubdiv")
	preserveVertexOrdering_ : PreserveVertexOrderingPlug = PlugDescriptor("preserveVertexOrdering")
	quickConvert_ : QuickConvertPlug = PlugDescriptor("quickConvert")
	uvTreatment_ : UvTreatmentPlug = PlugDescriptor("uvTreatment")

	# node attributes

	typeName = "polyToSubdiv"
	apiTypeInt = 685
	apiTypeStr = "kPolyToSubdiv"
	typeIdInt = 1347634259
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["absolutePosition", "applyMatrixToResult", "binMembership", "uvPointsU", "uvPointsV", "uvPoints", "cachedUVs", "inMesh", "maxEdgesPerVert", "maxPolyCount", "outSubdiv", "preserveVertexOrdering", "quickConvert", "uvTreatment"]
	nodeLeafPlugs = ["absolutePosition", "applyMatrixToResult", "binMembership", "cachedUVs", "inMesh", "maxEdgesPerVert", "maxPolyCount", "outSubdiv", "preserveVertexOrdering", "quickConvert", "uvTreatment"]
	pass

