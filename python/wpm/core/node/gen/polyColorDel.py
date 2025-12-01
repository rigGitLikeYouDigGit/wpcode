

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
class ColorSetNamePlug(Plug):
	node : PolyColorDel = None
	pass
# endregion


# define node class
class PolyColorDel(PolyModifier):
	colorSetName_ : ColorSetNamePlug = PlugDescriptor("colorSetName")

	# node attributes

	typeName = "polyColorDel"
	apiTypeInt = 741
	apiTypeStr = "kPolyColorDel"
	typeIdInt = 1346585676
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["colorSetName"]
	nodeLeafPlugs = ["colorSetName"]
	pass

