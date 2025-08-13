

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
	node : CopyColorSet = None
	pass
class DstColorNamePlug(Plug):
	node : CopyColorSet = None
	pass
class InputGeometryPlug(Plug):
	node : CopyColorSet = None
	pass
class OutputGeometryPlug(Plug):
	node : CopyColorSet = None
	pass
class SrcColorSetNamePlug(Plug):
	node : CopyColorSet = None
	pass
# endregion


# define node class
class CopyColorSet(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	dstColorName_ : DstColorNamePlug = PlugDescriptor("dstColorName")
	inputGeometry_ : InputGeometryPlug = PlugDescriptor("inputGeometry")
	outputGeometry_ : OutputGeometryPlug = PlugDescriptor("outputGeometry")
	srcColorSetName_ : SrcColorSetNamePlug = PlugDescriptor("srcColorSetName")

	# node attributes

	typeName = "copyColorSet"
	apiTypeInt = 738
	apiTypeStr = "kCopyColorSet"
	typeIdInt = 1129333587
	MFnCls = om.MFnDependencyNode
	pass

