

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
	node : ApplyAbsFloatOverride = None
	pass
class OutPlug(Plug):
	node : ApplyAbsFloatOverride = None
	pass
class ValuePlug(Plug):
	node : ApplyAbsFloatOverride = None
	pass
# endregion


# define node class
class ApplyAbsFloatOverride(ApplyAbsOverride):
	original_ : OriginalPlug = PlugDescriptor("original")
	out_ : OutPlug = PlugDescriptor("out")
	value_ : ValuePlug = PlugDescriptor("value")

	# node attributes

	typeName = "applyAbsFloatOverride"
	typeIdInt = 1476395901
	nodeLeafClassAttrs = ["original", "out", "value"]
	nodeLeafPlugs = ["original", "out", "value"]
	pass

