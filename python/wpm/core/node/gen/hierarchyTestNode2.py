

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
HierarchyTestNode1 = retriever.getNodeCls("HierarchyTestNode1")
assert HierarchyTestNode1
if T.TYPE_CHECKING:
	from .. import HierarchyTestNode1

# add node doc



# region plug type defs
class N2level2MPlug(Plug):
	parent : N2level1CPlug = PlugDescriptor("n2level1C")
	node : HierarchyTestNode2 = None
	pass
class N2level2SPlug(Plug):
	parent : N2level1CPlug = PlugDescriptor("n2level1C")
	node : HierarchyTestNode2 = None
	pass
class N2level1CPlug(Plug):
	parent : N2compoundPlug = PlugDescriptor("n2compound")
	n2level2M_ : N2level2MPlug = PlugDescriptor("n2level2M")
	n2m2_ : N2level2MPlug = PlugDescriptor("n2level2M")
	n2level2S_ : N2level2SPlug = PlugDescriptor("n2level2S")
	n2s2_ : N2level2SPlug = PlugDescriptor("n2level2S")
	node : HierarchyTestNode2 = None
	pass
class N2level1MPlug(Plug):
	parent : N2compoundPlug = PlugDescriptor("n2compound")
	node : HierarchyTestNode2 = None
	pass
class N2level1SPlug(Plug):
	parent : N2compoundPlug = PlugDescriptor("n2compound")
	node : HierarchyTestNode2 = None
	pass
class N2compoundPlug(Plug):
	n2level1C_ : N2level1CPlug = PlugDescriptor("n2level1C")
	n2c1_ : N2level1CPlug = PlugDescriptor("n2level1C")
	n2level1M_ : N2level1MPlug = PlugDescriptor("n2level1M")
	n2m1_ : N2level1MPlug = PlugDescriptor("n2level1M")
	n2level1S_ : N2level1SPlug = PlugDescriptor("n2level1S")
	n2s1_ : N2level1SPlug = PlugDescriptor("n2level1S")
	node : HierarchyTestNode2 = None
	pass
class N2singlePlug(Plug):
	node : HierarchyTestNode2 = None
	pass
# endregion


# define node class
class HierarchyTestNode2(HierarchyTestNode1):
	n2level2M_ : N2level2MPlug = PlugDescriptor("n2level2M")
	n2level2S_ : N2level2SPlug = PlugDescriptor("n2level2S")
	n2level1C_ : N2level1CPlug = PlugDescriptor("n2level1C")
	n2level1M_ : N2level1MPlug = PlugDescriptor("n2level1M")
	n2level1S_ : N2level1SPlug = PlugDescriptor("n2level1S")
	n2compound_ : N2compoundPlug = PlugDescriptor("n2compound")
	n2single_ : N2singlePlug = PlugDescriptor("n2single")

	# node attributes

	typeName = "hierarchyTestNode2"
	typeIdInt = 1213484594
	pass

