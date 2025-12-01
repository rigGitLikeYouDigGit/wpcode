

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
	node : MakeNurbCube = None
	pass
class HeightRatioPlug(Plug):
	node : MakeNurbCube = None
	pass
class LengthRatioPlug(Plug):
	node : MakeNurbCube = None
	pass
class OutputSurface1Plug(Plug):
	node : MakeNurbCube = None
	pass
class OutputSurface2Plug(Plug):
	node : MakeNurbCube = None
	pass
class OutputSurface3Plug(Plug):
	node : MakeNurbCube = None
	pass
class OutputSurface4Plug(Plug):
	node : MakeNurbCube = None
	pass
class OutputSurface5Plug(Plug):
	node : MakeNurbCube = None
	pass
class PatchesUPlug(Plug):
	node : MakeNurbCube = None
	pass
class PatchesVPlug(Plug):
	node : MakeNurbCube = None
	pass
class WidthPlug(Plug):
	node : MakeNurbCube = None
	pass
# endregion


# define node class
class MakeNurbCube(Primitive):
	degree_ : DegreePlug = PlugDescriptor("degree")
	heightRatio_ : HeightRatioPlug = PlugDescriptor("heightRatio")
	lengthRatio_ : LengthRatioPlug = PlugDescriptor("lengthRatio")
	outputSurface1_ : OutputSurface1Plug = PlugDescriptor("outputSurface1")
	outputSurface2_ : OutputSurface2Plug = PlugDescriptor("outputSurface2")
	outputSurface3_ : OutputSurface3Plug = PlugDescriptor("outputSurface3")
	outputSurface4_ : OutputSurface4Plug = PlugDescriptor("outputSurface4")
	outputSurface5_ : OutputSurface5Plug = PlugDescriptor("outputSurface5")
	patchesU_ : PatchesUPlug = PlugDescriptor("patchesU")
	patchesV_ : PatchesVPlug = PlugDescriptor("patchesV")
	width_ : WidthPlug = PlugDescriptor("width")

	# node attributes

	typeName = "makeNurbCube"
	typeIdInt = 1313035586
	nodeLeafClassAttrs = ["degree", "heightRatio", "lengthRatio", "outputSurface1", "outputSurface2", "outputSurface3", "outputSurface4", "outputSurface5", "patchesU", "patchesV", "width"]
	nodeLeafPlugs = ["degree", "heightRatio", "lengthRatio", "outputSurface1", "outputSurface2", "outputSurface3", "outputSurface4", "outputSurface5", "patchesU", "patchesV", "width"]
	pass

