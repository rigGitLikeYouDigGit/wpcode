

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
NurbsDimShape = retriever.getNodeCls("NurbsDimShape")
assert NurbsDimShape
if T.TYPE_CHECKING:
	from .. import NurbsDimShape

# add node doc



# region plug type defs

# endregion


# define node class
class ParamDimension(NurbsDimShape):

	# node attributes

	typeName = "paramDimension"
	apiTypeInt = 275
	apiTypeStr = "kParamDimension"
	typeIdInt = 1380207950
	MFnCls = om.MFnDagNode
	pass

