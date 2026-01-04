

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	HierarchyTestNode2 = Catalogue.HierarchyTestNode2
else:
	from .. import retriever
	HierarchyTestNode2 = retriever.getNodeCls("HierarchyTestNode2")
	assert HierarchyTestNode2

# add node doc



# region plug type defs
class N3level2MPlug(Plug):
	parent : N3level1CPlug = PlugDescriptor("n3level1C")
	node : HierarchyTestNode3 = None
	pass
class N3level2SPlug(Plug):
	parent : N3level1CPlug = PlugDescriptor("n3level1C")
	node : HierarchyTestNode3 = None
	pass
class N3level1CPlug(Plug):
	parent : N3compoundPlug = PlugDescriptor("n3compound")
	n3level2M_ : N3level2MPlug = PlugDescriptor("n3level2M")
	n3m2_ : N3level2MPlug = PlugDescriptor("n3level2M")
	n3level2S_ : N3level2SPlug = PlugDescriptor("n3level2S")
	n3s2_ : N3level2SPlug = PlugDescriptor("n3level2S")
	node : HierarchyTestNode3 = None
	pass
class N3level1MPlug(Plug):
	parent : N3compoundPlug = PlugDescriptor("n3compound")
	node : HierarchyTestNode3 = None
	pass
class N3level1SPlug(Plug):
	parent : N3compoundPlug = PlugDescriptor("n3compound")
	node : HierarchyTestNode3 = None
	pass
class N3compoundPlug(Plug):
	n3level1C_ : N3level1CPlug = PlugDescriptor("n3level1C")
	n3c1_ : N3level1CPlug = PlugDescriptor("n3level1C")
	n3level1M_ : N3level1MPlug = PlugDescriptor("n3level1M")
	n3m1_ : N3level1MPlug = PlugDescriptor("n3level1M")
	n3level1S_ : N3level1SPlug = PlugDescriptor("n3level1S")
	n3s1_ : N3level1SPlug = PlugDescriptor("n3level1S")
	node : HierarchyTestNode3 = None
	pass
class N3singlePlug(Plug):
	node : HierarchyTestNode3 = None
	pass
# endregion


# define node class
class HierarchyTestNode3(HierarchyTestNode2):
	n3level2M_ : N3level2MPlug = PlugDescriptor("n3level2M")
	n3level2S_ : N3level2SPlug = PlugDescriptor("n3level2S")
	n3level1C_ : N3level1CPlug = PlugDescriptor("n3level1C")
	n3level1M_ : N3level1MPlug = PlugDescriptor("n3level1M")
	n3level1S_ : N3level1SPlug = PlugDescriptor("n3level1S")
	n3compound_ : N3compoundPlug = PlugDescriptor("n3compound")
	n3single_ : N3singlePlug = PlugDescriptor("n3single")

	# node attributes

	typeName = "hierarchyTestNode3"
	typeIdInt = 1213484595
	nodeLeafClassAttrs = ["n3level2M", "n3level2S", "n3level1C", "n3level1M", "n3level1S", "n3compound", "n3single"]
	nodeLeafPlugs = ["n3compound", "n3single"]
	pass

