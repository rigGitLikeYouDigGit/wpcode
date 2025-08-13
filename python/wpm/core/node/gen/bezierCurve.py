

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
NurbsCurve = retriever.getNodeCls("NurbsCurve")
assert NurbsCurve
if T.TYPE_CHECKING:
	from .. import NurbsCurve

# add node doc



# region plug type defs
class AnchorSmoothnessPlug(Plug):
	node : BezierCurve = None
	pass
class AnchorWeightingPlug(Plug):
	node : BezierCurve = None
	pass
# endregion


# define node class
class BezierCurve(NurbsCurve):
	anchorSmoothness_ : AnchorSmoothnessPlug = PlugDescriptor("anchorSmoothness")
	anchorWeighting_ : AnchorWeightingPlug = PlugDescriptor("anchorWeighting")

	# node attributes

	typeName = "bezierCurve"
	typeIdInt = 1111708246
	pass

