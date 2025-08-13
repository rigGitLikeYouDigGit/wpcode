

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
class CenterOnTilePlug(Plug):
	node : PolyNormalizeUV = None
	pass
class NormalizeDirectionPlug(Plug):
	node : PolyNormalizeUV = None
	pass
class NormalizeTypePlug(Plug):
	node : PolyNormalizeUV = None
	pass
class PreserveAspectRatioPlug(Plug):
	node : PolyNormalizeUV = None
	pass
# endregion


# define node class
class PolyNormalizeUV(PolyModifierUV):
	centerOnTile_ : CenterOnTilePlug = PlugDescriptor("centerOnTile")
	normalizeDirection_ : NormalizeDirectionPlug = PlugDescriptor("normalizeDirection")
	normalizeType_ : NormalizeTypePlug = PlugDescriptor("normalizeType")
	preserveAspectRatio_ : PreserveAspectRatioPlug = PlugDescriptor("preserveAspectRatio")

	# node attributes

	typeName = "polyNormalizeUV"
	apiTypeInt = 887
	apiTypeStr = "kPolyNormalizeUV"
	typeIdInt = 1347310934
	MFnCls = om.MFnDependencyNode
	pass

