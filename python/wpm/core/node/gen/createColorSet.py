

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
	node : CreateColorSet = None
	pass
class ClampedPlug(Plug):
	node : CreateColorSet = None
	pass
class ColorSetNamePlug(Plug):
	node : CreateColorSet = None
	pass
class InputGeometryPlug(Plug):
	node : CreateColorSet = None
	pass
class OutputGeometryPlug(Plug):
	node : CreateColorSet = None
	pass
class RepresentationPlug(Plug):
	node : CreateColorSet = None
	pass
# endregion


# define node class
class CreateColorSet(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	clamped_ : ClampedPlug = PlugDescriptor("clamped")
	colorSetName_ : ColorSetNamePlug = PlugDescriptor("colorSetName")
	inputGeometry_ : InputGeometryPlug = PlugDescriptor("inputGeometry")
	outputGeometry_ : OutputGeometryPlug = PlugDescriptor("outputGeometry")
	representation_ : RepresentationPlug = PlugDescriptor("representation")

	# node attributes

	typeName = "createColorSet"
	apiTypeInt = 736
	apiTypeStr = "kCreateColorSet"
	typeIdInt = 1129464659
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "clamped", "colorSetName", "inputGeometry", "outputGeometry", "representation"]
	nodeLeafPlugs = ["binMembership", "clamped", "colorSetName", "inputGeometry", "outputGeometry", "representation"]
	pass

