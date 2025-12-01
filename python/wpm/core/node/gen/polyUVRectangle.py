

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifier = retriever.getNodeCls("PolyModifier")
assert PolyModifier
if T.TYPE_CHECKING:
	from .. import PolyModifier

# add node doc



# region plug type defs
class UvSetNamePlug(Plug):
	node : PolyUVRectangle = None
	pass
# endregion


# define node class
class PolyUVRectangle(PolyModifier):
	uvSetName_ : UvSetNamePlug = PlugDescriptor("uvSetName")

	# node attributes

	typeName = "polyUVRectangle"
	apiTypeInt = 1070
	apiTypeStr = "kPolyUVRectangle"
	typeIdInt = 1347769938
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["uvSetName"]
	nodeLeafPlugs = ["uvSetName"]
	pass

