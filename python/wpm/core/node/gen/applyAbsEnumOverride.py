

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ApplyAbsOverride = retriever.getNodeCls("ApplyAbsOverride")
assert ApplyAbsOverride
if T.TYPE_CHECKING:
	from .. import ApplyAbsOverride

# add node doc



# region plug type defs
class OriginalPlug(Plug):
	node : ApplyAbsEnumOverride = None
	pass
class OutPlug(Plug):
	node : ApplyAbsEnumOverride = None
	pass
class ValuePlug(Plug):
	node : ApplyAbsEnumOverride = None
	pass
# endregion


# define node class
class ApplyAbsEnumOverride(ApplyAbsOverride):
	original_ : OriginalPlug = PlugDescriptor("original")
	out_ : OutPlug = PlugDescriptor("out")
	value_ : ValuePlug = PlugDescriptor("value")

	# node attributes

	typeName = "applyAbsEnumOverride"
	typeIdInt = 1476395916
	pass

