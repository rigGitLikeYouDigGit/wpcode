

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AbsOverride = retriever.getNodeCls("AbsOverride")
assert AbsOverride
if T.TYPE_CHECKING:
	from .. import AbsOverride

# add node doc



# region plug type defs
class ConnectionStrPlug(Plug):
	node : ConnectionOverride = None
	pass
# endregion


# define node class
class ConnectionOverride(AbsOverride):
	connectionStr_ : ConnectionStrPlug = PlugDescriptor("connectionStr")

	# node attributes

	typeName = "connectionOverride"
	typeIdInt = 1476395909
	pass

