

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
class MarkerManip(Manip3D):

	# node attributes

	typeName = "markerManip"
	apiTypeInt = 210
	apiTypeStr = "kMarkerManip"
	typeIdInt = 1431129409
	MFnCls = om.MFnManip3D
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

