

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
_BASE_ = retriever.getNodeCls("_BASE_")
assert _BASE_
if T.TYPE_CHECKING:
	from .. import _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : SimpleTestNode = None
	pass
class Level1S1Plug(Plug):
	parent : CompoundPlug = PlugDescriptor("compound")
	node : SimpleTestNode = None
	pass
class Level1S2Plug(Plug):
	parent : CompoundPlug = PlugDescriptor("compound")
	node : SimpleTestNode = None
	pass
class Level1S3Plug(Plug):
	parent : CompoundPlug = PlugDescriptor("compound")
	node : SimpleTestNode = None
	pass
class CompoundPlug(Plug):
	level1S1_ : Level1S1Plug = PlugDescriptor("level1S1")
	l1s1_ : Level1S1Plug = PlugDescriptor("level1S1")
	level1S2_ : Level1S2Plug = PlugDescriptor("level1S2")
	l1s2_ : Level1S2Plug = PlugDescriptor("level1S2")
	level1S3_ : Level1S3Plug = PlugDescriptor("level1S3")
	l1s3_ : Level1S3Plug = PlugDescriptor("level1S3")
	node : SimpleTestNode = None
	pass
class FlagPlug(Plug):
	node : SimpleTestNode = None
	pass
class SinglePlug(Plug):
	node : SimpleTestNode = None
	pass
# endregion


# define node class
class SimpleTestNode(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	level1S1_ : Level1S1Plug = PlugDescriptor("level1S1")
	level1S2_ : Level1S2Plug = PlugDescriptor("level1S2")
	level1S3_ : Level1S3Plug = PlugDescriptor("level1S3")
	compound_ : CompoundPlug = PlugDescriptor("compound")
	flag_ : FlagPlug = PlugDescriptor("flag")
	single_ : SinglePlug = PlugDescriptor("single")

	# node attributes

	typeName = "simpleTestNode"
	typeIdInt = 1398033988
	pass

