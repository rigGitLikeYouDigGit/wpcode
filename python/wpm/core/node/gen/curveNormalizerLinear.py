

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
CurveNormalizer = retriever.getNodeCls("CurveNormalizer")
assert CurveNormalizer
if T.TYPE_CHECKING:
	from .. import CurveNormalizer

# add node doc



# region plug type defs
class AnimInputPlug(Plug):
	node : CurveNormalizerLinear = None
	pass
class OutputPlug(Plug):
	node : CurveNormalizerLinear = None
	pass
# endregion


# define node class
class CurveNormalizerLinear(CurveNormalizer):
	animInput_ : AnimInputPlug = PlugDescriptor("animInput")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "curveNormalizerLinear"
	apiTypeInt = 1004
	apiTypeStr = "kCurveNormalizerLinear"
	typeIdInt = 1129206348
	MFnCls = om.MFnDependencyNode
	pass

