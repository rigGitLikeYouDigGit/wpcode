

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ChildNode = retriever.getNodeCls("ChildNode")
assert ChildNode
if T.TYPE_CHECKING:
	from .. import ChildNode

# add node doc



# region plug type defs
class EnabledPlug(Plug):
	node : LightItemBase = None
	pass
class IsolateSelectedPlug(Plug):
	node : LightItemBase = None
	pass
class NumIsolatedAncestorsPlug(Plug):
	node : LightItemBase = None
	pass
class NumIsolatedChildrenPlug(Plug):
	node : LightItemBase = None
	pass
class ParentEnabledPlug(Plug):
	node : LightItemBase = None
	pass
class ParentNumIsolatedChildrenPlug(Plug):
	node : LightItemBase = None
	pass
class SelfEnabledPlug(Plug):
	node : LightItemBase = None
	pass
# endregion


# define node class
class LightItemBase(ChildNode):
	enabled_ : EnabledPlug = PlugDescriptor("enabled")
	isolateSelected_ : IsolateSelectedPlug = PlugDescriptor("isolateSelected")
	numIsolatedAncestors_ : NumIsolatedAncestorsPlug = PlugDescriptor("numIsolatedAncestors")
	numIsolatedChildren_ : NumIsolatedChildrenPlug = PlugDescriptor("numIsolatedChildren")
	parentEnabled_ : ParentEnabledPlug = PlugDescriptor("parentEnabled")
	parentNumIsolatedChildren_ : ParentNumIsolatedChildrenPlug = PlugDescriptor("parentNumIsolatedChildren")
	selfEnabled_ : SelfEnabledPlug = PlugDescriptor("selfEnabled")

	# node attributes

	typeName = "lightItemBase"
	typeIdInt = 1476396000
	pass

