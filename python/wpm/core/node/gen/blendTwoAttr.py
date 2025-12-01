

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Blend = retriever.getNodeCls("Blend")
assert Blend
if T.TYPE_CHECKING:
	from .. import Blend

# add node doc



# region plug type defs
class AttributesBlenderPlug(Plug):
	node : BlendTwoAttr = None
	pass
# endregion


# define node class
class BlendTwoAttr(Blend):
	attributesBlender_ : AttributesBlenderPlug = PlugDescriptor("attributesBlender")

	# node attributes

	typeName = "blendTwoAttr"
	apiTypeInt = 28
	apiTypeStr = "kBlendTwoAttr"
	typeIdInt = 1094863922
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["attributesBlender"]
	nodeLeafPlugs = ["attributesBlender"]
	pass

