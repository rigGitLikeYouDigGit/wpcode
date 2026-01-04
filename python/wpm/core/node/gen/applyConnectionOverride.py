

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	ApplyOverride = Catalogue.ApplyOverride
else:
	from .. import retriever
	ApplyOverride = retriever.getNodeCls("ApplyOverride")
	assert ApplyOverride

# add node doc



# region plug type defs
class NextPlug(Plug):
	node : ApplyConnectionOverride = None
	pass
class PreviousPlug(Plug):
	node : ApplyConnectionOverride = None
	pass
class TargetPlug(Plug):
	node : ApplyConnectionOverride = None
	pass
# endregion


# define node class
class ApplyConnectionOverride(ApplyOverride):
	next_ : NextPlug = PlugDescriptor("next")
	previous_ : PreviousPlug = PlugDescriptor("previous")
	target_ : TargetPlug = PlugDescriptor("target")

	# node attributes

	typeName = "applyConnectionOverride"
	typeIdInt = 1476395908
	nodeLeafClassAttrs = ["next", "previous", "target"]
	nodeLeafPlugs = ["next", "previous", "target"]
	pass

