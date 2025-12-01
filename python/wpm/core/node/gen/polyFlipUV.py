

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifierUV = retriever.getNodeCls("PolyModifierUV")
assert PolyModifierUV
if T.TYPE_CHECKING:
	from .. import PolyModifierUV

# add node doc



# region plug type defs
class CutUVPlug(Plug):
	node : PolyFlipUV = None
	pass
class FlipTypePlug(Plug):
	node : PolyFlipUV = None
	pass
class LocalPlug(Plug):
	node : PolyFlipUV = None
	pass
class PivotUPlug(Plug):
	node : PolyFlipUV = None
	pass
class PivotVPlug(Plug):
	node : PolyFlipUV = None
	pass
class UsePivotPlug(Plug):
	node : PolyFlipUV = None
	pass
# endregion


# define node class
class PolyFlipUV(PolyModifierUV):
	cutUV_ : CutUVPlug = PlugDescriptor("cutUV")
	flipType_ : FlipTypePlug = PlugDescriptor("flipType")
	local_ : LocalPlug = PlugDescriptor("local")
	pivotU_ : PivotUPlug = PlugDescriptor("pivotU")
	pivotV_ : PivotVPlug = PlugDescriptor("pivotV")
	usePivot_ : UsePivotPlug = PlugDescriptor("usePivot")

	# node attributes

	typeName = "polyFlipUV"
	apiTypeInt = 888
	apiTypeStr = "kPolyFlipUV"
	typeIdInt = 1346786646
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["cutUV", "flipType", "local", "pivotU", "pivotV", "usePivot"]
	nodeLeafPlugs = ["cutUV", "flipType", "local", "pivotU", "pivotV", "usePivot"]
	pass

