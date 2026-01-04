

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
class AttribAPlug(Plug):
	node : AboutToSetValueTestNode = None
	pass
class AttribBPlug(Plug):
	node : AboutToSetValueTestNode = None
	pass
class BinMembershipPlug(Plug):
	node : AboutToSetValueTestNode = None
	pass
# endregion


# define node class
class AboutToSetValueTestNode(_BASE_):
	attribA_ : AttribAPlug = PlugDescriptor("attribA")
	attribB_ : AttribBPlug = PlugDescriptor("attribB")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")

	# node attributes

	typeName = "aboutToSetValueTestNode"
	typeIdInt = 1095980622
	nodeLeafClassAttrs = ["attribA", "attribB", "binMembership"]
	nodeLeafPlugs = ["attribA", "attribB", "binMembership"]
	pass

