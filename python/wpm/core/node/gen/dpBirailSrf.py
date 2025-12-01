

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
class BlendFactorPlug(Plug):
	node : DpBirailSrf = None
	pass
class InputProfile1Plug(Plug):
	node : DpBirailSrf = None
	pass
class InputProfile2Plug(Plug):
	node : DpBirailSrf = None
	pass
class TangentContinuityProfile1Plug(Plug):
	node : DpBirailSrf = None
	pass
class TangentContinuityProfile2Plug(Plug):
	node : DpBirailSrf = None
	pass
# endregion


# define node class
class DpBirailSrf(BirailSrf):
	blendFactor_ : BlendFactorPlug = PlugDescriptor("blendFactor")
	inputProfile1_ : InputProfile1Plug = PlugDescriptor("inputProfile1")
	inputProfile2_ : InputProfile2Plug = PlugDescriptor("inputProfile2")
	tangentContinuityProfile1_ : TangentContinuityProfile1Plug = PlugDescriptor("tangentContinuityProfile1")
	tangentContinuityProfile2_ : TangentContinuityProfile2Plug = PlugDescriptor("tangentContinuityProfile2")

	# node attributes

	typeName = "dpBirailSrf"
	typeIdInt = 1313096275
	nodeLeafClassAttrs = ["blendFactor", "inputProfile1", "inputProfile2", "tangentContinuityProfile1", "tangentContinuityProfile2"]
	nodeLeafPlugs = ["blendFactor", "inputProfile1", "inputProfile2", "tangentContinuityProfile1", "tangentContinuityProfile2"]
	pass

