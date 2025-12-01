

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifier = retriever.getNodeCls("PolyModifier")
assert PolyModifier
if T.TYPE_CHECKING:
	from .. import PolyModifier

# add node doc



# region plug type defs
class EdgePlug(Plug):
	node : PolyFlipEdge = None
	pass
# endregion


# define node class
class PolyFlipEdge(PolyModifier):
	edge_ : EdgePlug = PlugDescriptor("edge")

	# node attributes

	typeName = "polyFlipEdge"
	apiTypeInt = 792
	apiTypeStr = "kPolyFlipEdge"
	typeIdInt = 1346784325
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["edge"]
	nodeLeafPlugs = ["edge"]
	pass

