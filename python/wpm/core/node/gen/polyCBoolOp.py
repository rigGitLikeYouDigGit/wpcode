

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyBoolOp = retriever.getNodeCls("PolyBoolOp")
assert PolyBoolOp
if T.TYPE_CHECKING:
	from .. import PolyBoolOp

# add node doc



# region plug type defs
class EdgeInterpolationPlug(Plug):
	node : PolyCBoolOp = None
	pass
class MergeGroupsPlug(Plug):
	node : PolyCBoolOp = None
	pass
class PlanarTolerancePlug(Plug):
	node : PolyCBoolOp = None
	pass
class SortOutputPlug(Plug):
	node : PolyCBoolOp = None
	pass
class TagIntersectionPlug(Plug):
	node : PolyCBoolOp = None
	pass
class UseCarveBooleansPlug(Plug):
	node : PolyCBoolOp = None
	pass
# endregion


# define node class
class PolyCBoolOp(PolyBoolOp):
	edgeInterpolation_ : EdgeInterpolationPlug = PlugDescriptor("edgeInterpolation")
	mergeGroups_ : MergeGroupsPlug = PlugDescriptor("mergeGroups")
	planarTolerance_ : PlanarTolerancePlug = PlugDescriptor("planarTolerance")
	sortOutput_ : SortOutputPlug = PlugDescriptor("sortOutput")
	tagIntersection_ : TagIntersectionPlug = PlugDescriptor("tagIntersection")
	useCarveBooleans_ : UseCarveBooleansPlug = PlugDescriptor("useCarveBooleans")

	# node attributes

	typeName = "polyCBoolOp"
	apiTypeInt = 1099
	apiTypeStr = "kPolyCBoolOp"
	typeIdInt = 1346590274
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["edgeInterpolation", "mergeGroups", "planarTolerance", "sortOutput", "tagIntersection", "useCarveBooleans"]
	nodeLeafPlugs = ["edgeInterpolation", "mergeGroups", "planarTolerance", "sortOutput", "tagIntersection", "useCarveBooleans"]
	pass

