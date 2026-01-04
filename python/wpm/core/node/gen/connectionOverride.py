

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	AbsOverride = Catalogue.AbsOverride
else:
	from .. import retriever
	AbsOverride = retriever.getNodeCls("AbsOverride")
	assert AbsOverride

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
	nodeLeafClassAttrs = ["connectionStr"]
	nodeLeafPlugs = ["connectionStr"]
	pass

