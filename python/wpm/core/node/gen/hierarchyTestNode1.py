

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
	node : HierarchyTestNode1 = None
	pass
class N1level2MPlug(Plug):
	parent : N1level1CPlug = PlugDescriptor("n1level1C")
	node : HierarchyTestNode1 = None
	pass
class N1level2SPlug(Plug):
	parent : N1level1CPlug = PlugDescriptor("n1level1C")
	node : HierarchyTestNode1 = None
	pass
class N1level1CPlug(Plug):
	parent : N1compoundPlug = PlugDescriptor("n1compound")
	n1level2M_ : N1level2MPlug = PlugDescriptor("n1level2M")
	n1m2_ : N1level2MPlug = PlugDescriptor("n1level2M")
	n1level2S_ : N1level2SPlug = PlugDescriptor("n1level2S")
	n1s2_ : N1level2SPlug = PlugDescriptor("n1level2S")
	node : HierarchyTestNode1 = None
	pass
class N1level1MPlug(Plug):
	parent : N1compoundPlug = PlugDescriptor("n1compound")
	node : HierarchyTestNode1 = None
	pass
class N1level1SPlug(Plug):
	parent : N1compoundPlug = PlugDescriptor("n1compound")
	node : HierarchyTestNode1 = None
	pass
class N1compoundPlug(Plug):
	n1level1C_ : N1level1CPlug = PlugDescriptor("n1level1C")
	n1c1_ : N1level1CPlug = PlugDescriptor("n1level1C")
	n1level1M_ : N1level1MPlug = PlugDescriptor("n1level1M")
	n1m1_ : N1level1MPlug = PlugDescriptor("n1level1M")
	n1level1S_ : N1level1SPlug = PlugDescriptor("n1level1S")
	n1s1_ : N1level1SPlug = PlugDescriptor("n1level1S")
	node : HierarchyTestNode1 = None
	pass
class N1singlePlug(Plug):
	node : HierarchyTestNode1 = None
	pass
# endregion


# define node class
class HierarchyTestNode1(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	n1level2M_ : N1level2MPlug = PlugDescriptor("n1level2M")
	n1level2S_ : N1level2SPlug = PlugDescriptor("n1level2S")
	n1level1C_ : N1level1CPlug = PlugDescriptor("n1level1C")
	n1level1M_ : N1level1MPlug = PlugDescriptor("n1level1M")
	n1level1S_ : N1level1SPlug = PlugDescriptor("n1level1S")
	n1compound_ : N1compoundPlug = PlugDescriptor("n1compound")
	n1single_ : N1singlePlug = PlugDescriptor("n1single")

	# node attributes

	typeName = "hierarchyTestNode1"
	typeIdInt = 1213484593
	nodeLeafClassAttrs = ["binMembership", "n1level2M", "n1level2S", "n1level1C", "n1level1M", "n1level1S", "n1compound", "n1single"]
	nodeLeafPlugs = ["binMembership", "n1compound", "n1single"]
	pass

