

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	THmanipContainer = Catalogue.THmanipContainer
else:
	from .. import retriever
	THmanipContainer = retriever.getNodeCls("THmanipContainer")
	assert THmanipContainer

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

