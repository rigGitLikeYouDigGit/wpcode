

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Manip3D = retriever.getNodeCls("Manip3D")
assert Manip3D
if T.TYPE_CHECKING:
	from .. import Manip3D

# add node doc



# region plug type defs

# endregion


# define node class
class AxesActionManip(Manip3D):

	# node attributes

	typeName = "axesActionManip"
	apiTypeInt = 124
	apiTypeStr = "kAxesActionManip"
	typeIdInt = 1430345793
	MFnCls = om.MFnManip3D
	pass

