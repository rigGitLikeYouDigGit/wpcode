

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
class ManipContainer(Manip3D):

	# node attributes

	typeName = "manipContainer"
	apiTypeInt = 148
	apiTypeStr = "kManipContainer"
	typeIdInt = 1431126851
	MFnCls = om.MFnManip3D
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

