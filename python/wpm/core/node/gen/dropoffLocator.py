

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Locator = Catalogue.Locator
else:
	from .. import retriever
	Locator = retriever.getNodeCls("Locator")
	assert Locator

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
	nodeLeafClassAttrs = ["param", "percent"]
	nodeLeafPlugs = ["param", "percent"]
	pass

