

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Locator = retriever.getNodeCls("Locator")
assert Locator
if T.TYPE_CHECKING:
	from .. import Locator

# add node doc



# region plug type defs
class ParamPlug(Plug):
	node : DropoffLocator = None
	pass
class PercentPlug(Plug):
	node : DropoffLocator = None
	pass
# endregion


# define node class
class DropoffLocator(Locator):
	param_ : ParamPlug = PlugDescriptor("param")
	percent_ : PercentPlug = PlugDescriptor("percent")

	# node attributes

	typeName = "dropoffLocator"
	apiTypeInt = 282
	apiTypeStr = "kDropoffLocator"
	typeIdInt = 1145848660
	MFnCls = om.MFnDagNode
	pass

