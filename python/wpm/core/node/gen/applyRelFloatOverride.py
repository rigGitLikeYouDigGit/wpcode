

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ApplyRelOverride = Catalogue.ApplyRelOverride
else:
	from .. import retriever
	ApplyRelOverride = retriever.getNodeCls("ApplyRelOverride")
	assert ApplyRelOverride

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
	nodeLeafClassAttrs = ["multiply", "offset", "original", "out"]
	nodeLeafPlugs = ["multiply", "offset", "original", "out"]
	pass

