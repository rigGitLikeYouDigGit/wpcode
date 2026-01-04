

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifier = Catalogue.PolyModifier
else:
	from .. import retriever
	PolyModifier = retriever.getNodeCls("PolyModifier")
	assert PolyModifier

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

