

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
class ColorSetNamePlug(Plug):
	node : PolyColorDel = None
	pass
# endregion


# define node class
class PolyColorDel(PolyModifier):
	colorSetName_ : ColorSetNamePlug = PlugDescriptor("colorSetName")

	# node attributes

	typeName = "polyColorDel"
	apiTypeInt = 741
	apiTypeStr = "kPolyColorDel"
	typeIdInt = 1346585676
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["colorSetName"]
	nodeLeafPlugs = ["colorSetName"]
	pass

