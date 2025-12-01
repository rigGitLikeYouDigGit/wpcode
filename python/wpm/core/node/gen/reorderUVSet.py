

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
	node : ReorderUVSet = None
	pass
class InputGeometryPlug(Plug):
	node : ReorderUVSet = None
	pass
class OutputGeometryPlug(Plug):
	node : ReorderUVSet = None
	pass
class UvSetNamePlug(Plug):
	node : ReorderUVSet = None
	pass
class UvSetName2Plug(Plug):
	node : ReorderUVSet = None
	pass
# endregion


# define node class
class ReorderUVSet(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	inputGeometry_ : InputGeometryPlug = PlugDescriptor("inputGeometry")
	outputGeometry_ : OutputGeometryPlug = PlugDescriptor("outputGeometry")
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")
	uvSetName2_ : UvSetName2Plug = PlugDescriptor("uvSetName2")

	# node attributes

	typeName = "reorderUVSet"
	apiTypeInt = 1133
	apiTypeStr = "kReorderUVSet"
	typeIdInt = 1380930899
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "inputGeometry", "outputGeometry", "uvSetName", "uvSetName2"]
	nodeLeafPlugs = ["binMembership", "inputGeometry", "outputGeometry", "uvSetName", "uvSetName2"]
	pass

