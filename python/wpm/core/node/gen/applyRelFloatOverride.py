

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
	node : ApplyRelFloatOverride = None
	pass
class OffsetPlug(Plug):
	node : ApplyRelFloatOverride = None
	pass
class OriginalPlug(Plug):
	node : ApplyRelFloatOverride = None
	pass
class OutPlug(Plug):
	node : ApplyRelFloatOverride = None
	pass
# endregion


# define node class
class ApplyRelFloatOverride(ApplyRelOverride):
	multiply_ : MultiplyPlug = PlugDescriptor("multiply")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	original_ : OriginalPlug = PlugDescriptor("original")
	out_ : OutPlug = PlugDescriptor("out")

	# node attributes

	typeName = "applyRelFloatOverride"
	typeIdInt = 1476395903
	pass

