

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
class IncludeHierarchyPlug(Plug):
	node : BasicSelector = None
	pass
# endregion


# define node class
class BasicSelector(SimpleSelector):
	includeHierarchy_ : IncludeHierarchyPlug = PlugDescriptor("includeHierarchy")

	# node attributes

	typeName = "basicSelector"
	typeIdInt = 1476395893
	nodeLeafClassAttrs = ["includeHierarchy"]
	nodeLeafPlugs = ["includeHierarchy"]
	pass

