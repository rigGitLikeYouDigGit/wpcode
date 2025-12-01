

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
THmanipContainer = retriever.getNodeCls("THmanipContainer")
assert THmanipContainer
if T.TYPE_CHECKING:
	from .. import THmanipContainer

# add node doc



# region plug type defs

# endregion


# define node class
class NexManip(THmanipContainer):

	# node attributes

	typeName = "nexManip"
	typeIdInt = 717856
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

