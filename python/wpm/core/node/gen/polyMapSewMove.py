

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
class LimitPieceSizePlug(Plug):
	node : PolyMapSewMove = None
	pass
class NumberFacesPlug(Plug):
	node : PolyMapSewMove = None
	pass
# endregion


# define node class
class PolyMapSewMove(PolyModifierUV):
	limitPieceSize_ : LimitPieceSizePlug = PlugDescriptor("limitPieceSize")
	numberFaces_ : NumberFacesPlug = PlugDescriptor("numberFaces")

	# node attributes

	typeName = "polyMapSewMove"
	apiTypeInt = 853
	apiTypeStr = "kPolyMapSewMove"
	typeIdInt = 1347634509
	MFnCls = om.MFnDependencyNode
	pass

