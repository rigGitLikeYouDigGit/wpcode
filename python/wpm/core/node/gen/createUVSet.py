

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
	node : CreateUVSet = None
	pass
class InputGeometryPlug(Plug):
	node : CreateUVSet = None
	pass
class OutputGeometryPlug(Plug):
	node : CreateUVSet = None
	pass
class UvSetNamePlug(Plug):
	node : CreateUVSet = None
	pass
# endregion


# define node class
class CreateUVSet(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	inputGeometry_ : InputGeometryPlug = PlugDescriptor("inputGeometry")
	outputGeometry_ : OutputGeometryPlug = PlugDescriptor("outputGeometry")
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")

	# node attributes

	typeName = "createUVSet"
	apiTypeInt = 808
	apiTypeStr = "kCreateUVSet"
	typeIdInt = 1129469267
	MFnCls = om.MFnDependencyNode
	pass

