

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
	node : AnimBlendNodeAdditiveFA = None
	pass
class InputBPlug(Plug):
	node : AnimBlendNodeAdditiveFA = None
	pass
class InterpolateModePlug(Plug):
	node : AnimBlendNodeAdditiveFA = None
	pass
class OutputPlug(Plug):
	node : AnimBlendNodeAdditiveFA = None
	pass
# endregion


# define node class
class AnimBlendNodeAdditiveFA(AnimBlendNodeBase):
	inputA_ : InputAPlug = PlugDescriptor("inputA")
	inputB_ : InputBPlug = PlugDescriptor("inputB")
	interpolateMode_ : InterpolateModePlug = PlugDescriptor("interpolateMode")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "animBlendNodeAdditiveFA"
	typeIdInt = 1094862401
	nodeLeafClassAttrs = ["inputA", "inputB", "interpolateMode", "output"]
	nodeLeafPlugs = ["inputA", "inputB", "interpolateMode", "output"]
	pass

