

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
BirailSrf = retriever.getNodeCls("BirailSrf")
assert BirailSrf
if T.TYPE_CHECKING:
	from .. import BirailSrf

# add node doc



# region plug type defs
class InputProfilePlug(Plug):
	node : SpBirailSrf = None
	pass
class TangentContinuityProfile1Plug(Plug):
	node : SpBirailSrf = None
	pass
# endregion


# define node class
class SpBirailSrf(BirailSrf):
	inputProfile_ : InputProfilePlug = PlugDescriptor("inputProfile")
	tangentContinuityProfile1_ : TangentContinuityProfile1Plug = PlugDescriptor("tangentContinuityProfile1")

	# node attributes

	typeName = "spBirailSrf"
	typeIdInt = 1314079315
	nodeLeafClassAttrs = ["inputProfile", "tangentContinuityProfile1"]
	nodeLeafPlugs = ["inputProfile", "tangentContinuityProfile1"]
	pass

