

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
class BlendOriginalPlug(Plug):
	node : PolyStraightenUVBorder = None
	pass
class CurvaturePlug(Plug):
	node : PolyStraightenUVBorder = None
	pass
class GapTolerancePlug(Plug):
	node : PolyStraightenUVBorder = None
	pass
class PreserveLengthPlug(Plug):
	node : PolyStraightenUVBorder = None
	pass
# endregion


# define node class
class PolyStraightenUVBorder(PolyModifierUV):
	blendOriginal_ : BlendOriginalPlug = PlugDescriptor("blendOriginal")
	curvature_ : CurvaturePlug = PlugDescriptor("curvature")
	gapTolerance_ : GapTolerancePlug = PlugDescriptor("gapTolerance")
	preserveLength_ : PreserveLengthPlug = PlugDescriptor("preserveLength")

	# node attributes

	typeName = "polyStraightenUVBorder"
	apiTypeInt = 911
	apiTypeStr = "kPolyStraightenUVBorder"
	typeIdInt = 1347638338
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["blendOriginal", "curvature", "gapTolerance", "preserveLength"]
	nodeLeafPlugs = ["blendOriginal", "curvature", "gapTolerance", "preserveLength"]
	pass

