

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyMoveFace = retriever.getNodeCls("PolyMoveFace")
assert PolyMoveFace
if T.TYPE_CHECKING:
	from .. import PolyMoveFace

# add node doc



# region plug type defs
class DuplicatePlug(Plug):
	node : PolyChipOff = None
	pass
class KeepFacesTogetherPlug(Plug):
	node : PolyChipOff = None
	pass
# endregion


# define node class
class PolyChipOff(PolyMoveFace):
	duplicate_ : DuplicatePlug = PlugDescriptor("duplicate")
	keepFacesTogether_ : KeepFacesTogetherPlug = PlugDescriptor("keepFacesTogether")

	# node attributes

	typeName = "polyChipOff"
	apiTypeInt = 404
	apiTypeStr = "kPolyChipOff"
	typeIdInt = 1346586697
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["duplicate", "keepFacesTogether"]
	nodeLeafPlugs = ["duplicate", "keepFacesTogether"]
	pass

