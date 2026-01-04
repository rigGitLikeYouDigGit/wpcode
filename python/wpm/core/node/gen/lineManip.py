

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Manip3D = Catalogue.Manip3D
else:
	from .. import retriever
	Manip3D = retriever.getNodeCls("Manip3D")
	assert Manip3D

# add node doc



# region plug type defs

# endregion


# define node class
class LineManip(Manip3D):

	# node attributes

	typeName = "lineManip"
	apiTypeInt = 147
	apiTypeStr = "kLineManip"
	typeIdInt = 1431129166
	MFnCls = om.MFnManip3D
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

