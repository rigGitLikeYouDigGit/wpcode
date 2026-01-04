

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ApplyAbsOverride = Catalogue.ApplyAbsOverride
else:
	from .. import retriever
	ApplyAbsOverride = retriever.getNodeCls("ApplyAbsOverride")
	assert ApplyAbsOverride

# add node doc



# region plug type defs
class OriginalPlug(Plug):
	node : ApplyAbsIntOverride = None
	pass
class OutPlug(Plug):
	node : ApplyAbsIntOverride = None
	pass
class ValuePlug(Plug):
	node : ApplyAbsIntOverride = None
	pass
# endregion


# define node class
class ApplyAbsIntOverride(ApplyAbsOverride):
	original_ : OriginalPlug = PlugDescriptor("original")
	out_ : OutPlug = PlugDescriptor("out")
	value_ : ValuePlug = PlugDescriptor("value")

	# node attributes

	typeName = "applyAbsIntOverride"
	typeIdInt = 1476395921
	nodeLeafClassAttrs = ["original", "out", "value"]
	nodeLeafPlugs = ["original", "out", "value"]
	pass

