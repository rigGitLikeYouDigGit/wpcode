

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AnimBlendNodeBase = retriever.getNodeCls("AnimBlendNodeBase")
assert AnimBlendNodeBase
if T.TYPE_CHECKING:
	from .. import AnimBlendNodeBase

# add node doc



# region plug type defs
class InputAPlug(Plug):
	node : AnimBlendNodeAdditiveDA = None
	pass
class InputBPlug(Plug):
	node : AnimBlendNodeAdditiveDA = None
	pass
class InterpolateModePlug(Plug):
	node : AnimBlendNodeAdditiveDA = None
	pass
class OutputPlug(Plug):
	node : AnimBlendNodeAdditiveDA = None
	pass
# endregion


# define node class
class AnimBlendNodeAdditiveDA(AnimBlendNodeBase):
	inputA_ : InputAPlug = PlugDescriptor("inputA")
	inputB_ : InputBPlug = PlugDescriptor("inputB")
	interpolateMode_ : InterpolateModePlug = PlugDescriptor("interpolateMode")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "animBlendNodeAdditiveDA"
	typeIdInt = 1094861121
	pass

