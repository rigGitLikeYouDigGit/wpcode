

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
DeformFunc = retriever.getNodeCls("DeformFunc")
assert DeformFunc
if T.TYPE_CHECKING:
	from .. import DeformFunc

# add node doc



# region plug type defs
class CurvePlug(Plug):
	node : DeformFlare = None
	pass
class EndFlareXPlug(Plug):
	node : DeformFlare = None
	pass
class EndFlareZPlug(Plug):
	node : DeformFlare = None
	pass
class HighBoundPlug(Plug):
	node : DeformFlare = None
	pass
class LowBoundPlug(Plug):
	node : DeformFlare = None
	pass
class StartFlareXPlug(Plug):
	node : DeformFlare = None
	pass
class StartFlareZPlug(Plug):
	node : DeformFlare = None
	pass
# endregion


# define node class
class DeformFlare(DeformFunc):
	curve_ : CurvePlug = PlugDescriptor("curve")
	endFlareX_ : EndFlareXPlug = PlugDescriptor("endFlareX")
	endFlareZ_ : EndFlareZPlug = PlugDescriptor("endFlareZ")
	highBound_ : HighBoundPlug = PlugDescriptor("highBound")
	lowBound_ : LowBoundPlug = PlugDescriptor("lowBound")
	startFlareX_ : StartFlareXPlug = PlugDescriptor("startFlareX")
	startFlareZ_ : StartFlareZPlug = PlugDescriptor("startFlareZ")

	# node attributes

	typeName = "deformFlare"
	apiTypeInt = 628
	apiTypeStr = "kDeformFlare"
	typeIdInt = 1178879564
	MFnCls = om.MFnDagNode
	pass

