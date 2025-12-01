

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Ffd = retriever.getNodeCls("Ffd")
assert Ffd
if T.TYPE_CHECKING:
	from .. import Ffd

# add node doc



# region plug type defs
class BaseLattice2MatrixPlug(Plug):
	node : JointFfd = None
	pass
class GroupIdLowerBindSkinPlug(Plug):
	node : JointFfd = None
	pass
class GroupIdUpperBindSkinPlug(Plug):
	node : JointFfd = None
	pass
class LowerBindSkinNodePlug(Plug):
	node : JointFfd = None
	pass
class LowerComponentCachePlug(Plug):
	node : JointFfd = None
	pass
class UpperBindSkinNodePlug(Plug):
	node : JointFfd = None
	pass
class UpperComponentCachePlug(Plug):
	node : JointFfd = None
	pass
class UseComponentCachePlug(Plug):
	node : JointFfd = None
	pass
# endregion


# define node class
class JointFfd(Ffd):
	baseLattice2Matrix_ : BaseLattice2MatrixPlug = PlugDescriptor("baseLattice2Matrix")
	groupIdLowerBindSkin_ : GroupIdLowerBindSkinPlug = PlugDescriptor("groupIdLowerBindSkin")
	groupIdUpperBindSkin_ : GroupIdUpperBindSkinPlug = PlugDescriptor("groupIdUpperBindSkin")
	lowerBindSkinNode_ : LowerBindSkinNodePlug = PlugDescriptor("lowerBindSkinNode")
	lowerComponentCache_ : LowerComponentCachePlug = PlugDescriptor("lowerComponentCache")
	upperBindSkinNode_ : UpperBindSkinNodePlug = PlugDescriptor("upperBindSkinNode")
	upperComponentCache_ : UpperComponentCachePlug = PlugDescriptor("upperComponentCache")
	useComponentCache_ : UseComponentCachePlug = PlugDescriptor("useComponentCache")

	# node attributes

	typeName = "jointFfd"
	typeIdInt = 1179010114
	nodeLeafClassAttrs = ["baseLattice2Matrix", "groupIdLowerBindSkin", "groupIdUpperBindSkin", "lowerBindSkinNode", "lowerComponentCache", "upperBindSkinNode", "upperComponentCache", "useComponentCache"]
	nodeLeafPlugs = ["baseLattice2Matrix", "groupIdLowerBindSkin", "groupIdUpperBindSkin", "lowerBindSkinNode", "lowerComponentCache", "upperBindSkinNode", "upperComponentCache", "useComponentCache"]
	pass

