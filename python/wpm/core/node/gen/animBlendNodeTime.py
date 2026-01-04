

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	AnimBlendNodeBase = Catalogue.AnimBlendNodeBase
else:
	from .. import retriever
	AnimBlendNodeBase = retriever.getNodeCls("AnimBlendNodeBase")
	assert AnimBlendNodeBase

# add node doc



# region plug type defs
class InputAPlug(Plug):
	node : AnimBlendNodeTime = None
	pass
class InputBPlug(Plug):
	node : AnimBlendNodeTime = None
	pass
class OutputPlug(Plug):
	node : AnimBlendNodeTime = None
	pass
# endregion


# define node class
class AnimBlendNodeTime(AnimBlendNodeBase):
	inputA_ : InputAPlug = PlugDescriptor("inputA")
	inputB_ : InputBPlug = PlugDescriptor("inputB")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "animBlendNodeTime"
	typeIdInt = 1094865993
	nodeLeafClassAttrs = ["inputA", "inputB", "output"]
	nodeLeafPlugs = ["inputA", "inputB", "output"]
	pass

