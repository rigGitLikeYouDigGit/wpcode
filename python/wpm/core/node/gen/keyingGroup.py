

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ObjectSet = retriever.getNodeCls("ObjectSet")
assert ObjectSet
if T.TYPE_CHECKING:
	from .. import ObjectSet

# add node doc



# region plug type defs
class ActivatorsPlug(Plug):
	node : KeyingGroup = None
	pass
class CategoryPlug(Plug):
	node : KeyingGroup = None
	pass
class MinimizeRotationPlug(Plug):
	node : KeyingGroup = None
	pass
# endregion


# define node class
class KeyingGroup(ObjectSet):
	activators_ : ActivatorsPlug = PlugDescriptor("activators")
	category_ : CategoryPlug = PlugDescriptor("category")
	minimizeRotation_ : MinimizeRotationPlug = PlugDescriptor("minimizeRotation")

	# node attributes

	typeName = "keyingGroup"
	apiTypeInt = 687
	apiTypeStr = "kKeyingGroup"
	typeIdInt = 1262965328
	MFnCls = om.MFnSet
	pass

