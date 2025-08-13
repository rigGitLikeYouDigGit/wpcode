

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
	node : MpBirailSrf = None
	pass
class TangentContinuityProfile1Plug(Plug):
	node : MpBirailSrf = None
	pass
class TangentContinuityProfile2Plug(Plug):
	node : MpBirailSrf = None
	pass
# endregion


# define node class
class MpBirailSrf(BirailSrf):
	inputProfile_ : InputProfilePlug = PlugDescriptor("inputProfile")
	tangentContinuityProfile1_ : TangentContinuityProfile1Plug = PlugDescriptor("tangentContinuityProfile1")
	tangentContinuityProfile2_ : TangentContinuityProfile2Plug = PlugDescriptor("tangentContinuityProfile2")

	# node attributes

	typeName = "mpBirailSrf"
	typeIdInt = 1313686099
	pass

