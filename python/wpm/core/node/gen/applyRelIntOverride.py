

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ApplyRelOverride = retriever.getNodeCls("ApplyRelOverride")
assert ApplyRelOverride
if T.TYPE_CHECKING:
	from .. import ApplyRelOverride

# add node doc



# region plug type defs
class MultiplyPlug(Plug):
	node : ApplyRelIntOverride = None
	pass
class OffsetPlug(Plug):
	node : ApplyRelIntOverride = None
	pass
class OriginalPlug(Plug):
	node : ApplyRelIntOverride = None
	pass
class OutPlug(Plug):
	node : ApplyRelIntOverride = None
	pass
# endregion


# define node class
class ApplyRelIntOverride(ApplyRelOverride):
	multiply_ : MultiplyPlug = PlugDescriptor("multiply")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	original_ : OriginalPlug = PlugDescriptor("original")
	out_ : OutPlug = PlugDescriptor("out")

	# node attributes

	typeName = "applyRelIntOverride"
	typeIdInt = 1476395922
	nodeLeafClassAttrs = ["multiply", "offset", "original", "out"]
	nodeLeafPlugs = ["multiply", "offset", "original", "out"]
	pass

