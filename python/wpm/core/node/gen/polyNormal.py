

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifier = Catalogue.PolyModifier
else:
	from .. import retriever
	PolyModifier = retriever.getNodeCls("PolyModifier")
	assert PolyModifier

# add node doc



# region plug type defs
class NormalModePlug(Plug):
	node : PolyNormal = None
	pass
class UserNormalModePlug(Plug):
	node : PolyNormal = None
	pass
# endregion


# define node class
class PolyNormal(PolyModifier):
	normalMode_ : NormalModePlug = PlugDescriptor("normalMode")
	userNormalMode_ : UserNormalModePlug = PlugDescriptor("userNormalMode")

	# node attributes

	typeName = "polyNormal"
	apiTypeInt = 424
	apiTypeStr = "kPolyNormal"
	typeIdInt = 1347309394
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["normalMode", "userNormalMode"]
	nodeLeafPlugs = ["normalMode", "userNormalMode"]
	pass

