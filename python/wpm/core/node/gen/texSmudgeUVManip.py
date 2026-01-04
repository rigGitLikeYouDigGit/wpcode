

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	TexBaseDeformManip = Catalogue.TexBaseDeformManip
else:
	from .. import retriever
	TexBaseDeformManip = retriever.getNodeCls("TexBaseDeformManip")
	assert TexBaseDeformManip

# add node doc



# region plug type defs

# endregion


# define node class
class TexSmudgeUVManip(TexBaseDeformManip):

	# node attributes

	typeName = "texSmudgeUVManip"
	apiTypeInt = 198
	apiTypeStr = "kTexSmudgeUVManip"
	typeIdInt = 1414745421
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

