

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
SubdModifierUV = retriever.getNodeCls("SubdModifierUV")
assert SubdModifierUV
if T.TYPE_CHECKING:
	from .. import SubdModifierUV

# add node doc



# region plug type defs
class LimitPieceSizePlug(Plug):
	node : SubdMapSewMove = None
	pass
class NumberFacesPlug(Plug):
	node : SubdMapSewMove = None
	pass
# endregion


# define node class
class SubdMapSewMove(SubdModifierUV):
	limitPieceSize_ : LimitPieceSizePlug = PlugDescriptor("limitPieceSize")
	numberFaces_ : NumberFacesPlug = PlugDescriptor("numberFaces")

	# node attributes

	typeName = "subdMapSewMove"
	apiTypeInt = 874
	apiTypeStr = "kSubdMapSewMove"
	typeIdInt = 1397966157
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["limitPieceSize", "numberFaces"]
	nodeLeafPlugs = ["limitPieceSize", "numberFaces"]
	pass

