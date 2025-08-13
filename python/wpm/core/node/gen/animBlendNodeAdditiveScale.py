

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
class AccumulationModePlug(Plug):
	node : AnimBlendNodeAdditiveScale = None
	pass
class InputAPlug(Plug):
	node : AnimBlendNodeAdditiveScale = None
	pass
class InputBPlug(Plug):
	node : AnimBlendNodeAdditiveScale = None
	pass
class OutputPlug(Plug):
	node : AnimBlendNodeAdditiveScale = None
	pass
# endregion


# define node class
class AnimBlendNodeAdditiveScale(AnimBlendNodeBase):
	accumulationMode_ : AccumulationModePlug = PlugDescriptor("accumulationMode")
	inputA_ : InputAPlug = PlugDescriptor("inputA")
	inputB_ : InputBPlug = PlugDescriptor("inputB")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "animBlendNodeAdditiveScale"
	typeIdInt = 1094864467
	pass

