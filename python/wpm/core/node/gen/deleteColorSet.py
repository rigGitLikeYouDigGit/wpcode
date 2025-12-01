

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
	node : DeleteColorSet = None
	pass
class ColorSetNamePlug(Plug):
	node : DeleteColorSet = None
	pass
class InputGeometryPlug(Plug):
	node : DeleteColorSet = None
	pass
class OutputGeometryPlug(Plug):
	node : DeleteColorSet = None
	pass
# endregion


# define node class
class DeleteColorSet(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	colorSetName_ : ColorSetNamePlug = PlugDescriptor("colorSetName")
	inputGeometry_ : InputGeometryPlug = PlugDescriptor("inputGeometry")
	outputGeometry_ : OutputGeometryPlug = PlugDescriptor("outputGeometry")

	# node attributes

	typeName = "deleteColorSet"
	apiTypeInt = 737
	apiTypeStr = "kDeleteColorSet"
	typeIdInt = 1145848659
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "colorSetName", "inputGeometry", "outputGeometry"]
	nodeLeafPlugs = ["binMembership", "colorSetName", "inputGeometry", "outputGeometry"]
	pass

