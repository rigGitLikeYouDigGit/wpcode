

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ManipContainer = retriever.getNodeCls("ManipContainer")
assert ManipContainer
if T.TYPE_CHECKING:
	from .. import ManipContainer

# add node doc



# region plug type defs

# endregion


# define node class
class DeformBendManip(ManipContainer):

	# node attributes

	typeName = "deformBendManip"
	apiTypeInt = 631
	apiTypeStr = "kDeformBendManip"
	typeIdInt = 1430408772
	MFnCls = om.MFnManip3D
	pass

