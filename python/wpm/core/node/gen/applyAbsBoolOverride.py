

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
	node : ApplyAbsBoolOverride = None
	pass
class OutPlug(Plug):
	node : ApplyAbsBoolOverride = None
	pass
class ValuePlug(Plug):
	node : ApplyAbsBoolOverride = None
	pass
# endregion


# define node class
class ApplyAbsBoolOverride(ApplyAbsOverride):
	original_ : OriginalPlug = PlugDescriptor("original")
	out_ : OutPlug = PlugDescriptor("out")
	value_ : ValuePlug = PlugDescriptor("value")

	# node attributes

	typeName = "applyAbsBoolOverride"
	typeIdInt = 1476395914
	nodeLeafClassAttrs = ["original", "out", "value"]
	nodeLeafPlugs = ["original", "out", "value"]
	pass

