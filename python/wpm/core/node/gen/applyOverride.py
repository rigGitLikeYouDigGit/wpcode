

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : ApplyOverride = None
	pass
class EnabledPlug(Plug):
	node : ApplyOverride = None
	pass
# endregion


# define node class
class ApplyOverride(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	enabled_ : EnabledPlug = PlugDescriptor("enabled")

	# node attributes

	typeName = "applyOverride"
	typeIdInt = 1476395895
	nodeLeafClassAttrs = ["binMembership", "enabled"]
	nodeLeafPlugs = ["binMembership", "enabled"]
	pass

