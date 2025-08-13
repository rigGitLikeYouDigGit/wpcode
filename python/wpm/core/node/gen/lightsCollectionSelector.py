

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
SimpleSelector = retriever.getNodeCls("SimpleSelector")
assert SimpleSelector
if T.TYPE_CHECKING:
	from .. import SimpleSelector

# add node doc



# region plug type defs

# endregion


# define node class
class LightsCollectionSelector(SimpleSelector):

	# node attributes

	typeName = "lightsCollectionSelector"
	typeIdInt = 1476395940
	pass

