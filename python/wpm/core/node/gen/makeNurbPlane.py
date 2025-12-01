

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Primitive = retriever.getNodeCls("Primitive")
assert Primitive
if T.TYPE_CHECKING:
	from .. import Primitive

# add node doc



# region plug type defs
class DegreePlug(Plug):
	node : MakeNurbPlane = None
	pass
class LengthRatioPlug(Plug):
	node : MakeNurbPlane = None
	pass
class PatchesUPlug(Plug):
	node : MakeNurbPlane = None
	pass
class PatchesVPlug(Plug):
	node : MakeNurbPlane = None
	pass
class WidthPlug(Plug):
	node : MakeNurbPlane = None
	pass
# endregion


# define node class
class MakeNurbPlane(Primitive):
	degree_ : DegreePlug = PlugDescriptor("degree")
	lengthRatio_ : LengthRatioPlug = PlugDescriptor("lengthRatio")
	patchesU_ : PatchesUPlug = PlugDescriptor("patchesU")
	patchesV_ : PatchesVPlug = PlugDescriptor("patchesV")
	width_ : WidthPlug = PlugDescriptor("width")

	# node attributes

	typeName = "makeNurbPlane"
	typeIdInt = 1313885262
	nodeLeafClassAttrs = ["degree", "lengthRatio", "patchesU", "patchesV", "width"]
	nodeLeafPlugs = ["degree", "lengthRatio", "patchesU", "patchesV", "width"]
	pass

