

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
CurveFromSubdiv = retriever.getNodeCls("CurveFromSubdiv")
assert CurveFromSubdiv
if T.TYPE_CHECKING:
	from .. import CurveFromSubdiv

# add node doc



# region plug type defs
class FaceIndexLPlug(Plug):
	node : CurveFromSubdivFace = None
	pass
class FaceIndexRPlug(Plug):
	node : CurveFromSubdivFace = None
	pass
# endregion


# define node class
class CurveFromSubdivFace(CurveFromSubdiv):
	faceIndexL_ : FaceIndexLPlug = PlugDescriptor("faceIndexL")
	faceIndexR_ : FaceIndexRPlug = PlugDescriptor("faceIndexR")

	# node attributes

	typeName = "curveFromSubdivFace"
	apiTypeInt = 842
	apiTypeStr = "kCurveFromSubdivFace"
	typeIdInt = 1396921158
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["faceIndexL", "faceIndexR"]
	nodeLeafPlugs = ["faceIndexL", "faceIndexR"]
	pass

