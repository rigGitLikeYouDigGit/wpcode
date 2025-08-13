

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Texture2d = retriever.getNodeCls("Texture2d")
assert Texture2d
if T.TYPE_CHECKING:
	from .. import Texture2d

# add node doc



# region plug type defs
class UWidthPlug(Plug):
	node : Bulge = None
	pass
class VWidthPlug(Plug):
	node : Bulge = None
	pass
# endregion


# define node class
class Bulge(Texture2d):
	uWidth_ : UWidthPlug = PlugDescriptor("uWidth")
	vWidth_ : VWidthPlug = PlugDescriptor("vWidth")

	# node attributes

	typeName = "bulge"
	apiTypeInt = 497
	apiTypeStr = "kBulge"
	typeIdInt = 1381253717
	MFnCls = om.MFnDependencyNode
	pass

