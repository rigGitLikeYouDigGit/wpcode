

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Transform = retriever.getNodeCls("Transform")
assert Transform
if T.TYPE_CHECKING:
	from .. import Transform

# add node doc



# region plug type defs
class EnableRestPositionPlug(Plug):
	node : Constraint = None
	pass
class LockOutputPlug(Plug):
	node : Constraint = None
	pass
# endregion


# define node class
class Constraint(Transform):
	enableRestPosition_ : EnableRestPositionPlug = PlugDescriptor("enableRestPosition")
	lockOutput_ : LockOutputPlug = PlugDescriptor("lockOutput")

	# node attributes

	typeName = "constraint"
	typeIdInt = 1129270867
	nodeLeafClassAttrs = ["enableRestPosition", "lockOutput"]
	nodeLeafPlugs = ["enableRestPosition", "lockOutput"]
	pass

