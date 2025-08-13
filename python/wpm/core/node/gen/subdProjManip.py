

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ProjectionManip = retriever.getNodeCls("ProjectionManip")
assert ProjectionManip
if T.TYPE_CHECKING:
	from .. import ProjectionManip

# add node doc



# region plug type defs

# endregion


# define node class
class SubdProjManip(ProjectionManip):

	# node attributes

	typeName = "subdProjManip"
	typeIdInt = 1431522641
	pass

