

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Override = retriever.getNodeCls("Override")
assert Override
if T.TYPE_CHECKING:
	from .. import Override

# add node doc



# region plug type defs

# endregion


# define node class
class ValueOverride(Override):

	# node attributes

	typeName = "valueOverride"
	typeIdInt = 1476395912
	pass

