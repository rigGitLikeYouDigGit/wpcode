

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
	node : AnimBlendNodeAdditiveF = None
	pass
class InputBPlug(Plug):
	node : AnimBlendNodeAdditiveF = None
	pass
class InterpolateModePlug(Plug):
	node : AnimBlendNodeAdditiveF = None
	pass
class OutputPlug(Plug):
	node : AnimBlendNodeAdditiveF = None
	pass
# endregion


# define node class
class AnimBlendNodeAdditiveF(AnimBlendNodeBase):
	inputA_ : InputAPlug = PlugDescriptor("inputA")
	inputB_ : InputBPlug = PlugDescriptor("inputB")
	interpolateMode_ : InterpolateModePlug = PlugDescriptor("interpolateMode")
	output_ : OutputPlug = PlugDescriptor("output")

	# node attributes

	typeName = "animBlendNodeAdditiveF"
	typeIdInt = 1094861126
	nodeLeafClassAttrs = ["inputA", "inputB", "interpolateMode", "output"]
	nodeLeafPlugs = ["inputA", "inputB", "interpolateMode", "output"]
	pass

