

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ListItem = retriever.getNodeCls("ListItem")
assert ListItem
if T.TYPE_CHECKING:
	from .. import ListItem

# add node doc



# region plug type defs
class ChildHighestPlug(Plug):
	node : RScontainer = None
	pass
class ChildLowestPlug(Plug):
	node : RScontainer = None
	pass
class EnabledPlug(Plug):
	node : RScontainer = None
	pass
class IsolateSelectedPlug(Plug):
	node : RScontainer = None
	pass
class ListItemsPlug(Plug):
	node : RScontainer = None
	pass
class NumIsolatedAncestorsPlug(Plug):
	node : RScontainer = None
	pass
class NumIsolatedChildrenPlug(Plug):
	node : RScontainer = None
	pass
class ParentEnabledPlug(Plug):
	node : RScontainer = None
	pass
class ParentNumIsolatedChildrenPlug(Plug):
	node : RScontainer = None
	pass
class SelfEnabledPlug(Plug):
	node : RScontainer = None
	pass
# endregion


# define node class
class RScontainer(ListItem):
	childHighest_ : ChildHighestPlug = PlugDescriptor("childHighest")
	childLowest_ : ChildLowestPlug = PlugDescriptor("childLowest")
	enabled_ : EnabledPlug = PlugDescriptor("enabled")
	isolateSelected_ : IsolateSelectedPlug = PlugDescriptor("isolateSelected")
	listItems_ : ListItemsPlug = PlugDescriptor("listItems")
	numIsolatedAncestors_ : NumIsolatedAncestorsPlug = PlugDescriptor("numIsolatedAncestors")
	numIsolatedChildren_ : NumIsolatedChildrenPlug = PlugDescriptor("numIsolatedChildren")
	parentEnabled_ : ParentEnabledPlug = PlugDescriptor("parentEnabled")
	parentNumIsolatedChildren_ : ParentNumIsolatedChildrenPlug = PlugDescriptor("parentNumIsolatedChildren")
	selfEnabled_ : SelfEnabledPlug = PlugDescriptor("selfEnabled")

	# node attributes

	typeName = "RScontainer"
	typeIdInt = 1476395942
	nodeLeafClassAttrs = ["childHighest", "childLowest", "enabled", "isolateSelected", "listItems", "numIsolatedAncestors", "numIsolatedChildren", "parentEnabled", "parentNumIsolatedChildren", "selfEnabled"]
	nodeLeafPlugs = ["childHighest", "childLowest", "enabled", "isolateSelected", "listItems", "numIsolatedAncestors", "numIsolatedChildren", "parentEnabled", "parentNumIsolatedChildren", "selfEnabled"]
	pass

