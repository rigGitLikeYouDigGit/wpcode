

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
DynBase = retriever.getNodeCls("DynBase")
assert DynBase
if T.TYPE_CHECKING:
	from .. import DynBase

# add node doc



# region plug type defs
class FrictionPlug(Plug):
	node : CollisionModel = None
	pass
class ResiliencePlug(Plug):
	node : CollisionModel = None
	pass
# endregion


# define node class
class CollisionModel(DynBase):
	friction_ : FrictionPlug = PlugDescriptor("friction")
	resilience_ : ResiliencePlug = PlugDescriptor("resilience")

	# node attributes

	typeName = "collisionModel"
	typeIdInt = 1497583436
	nodeLeafClassAttrs = ["friction", "resilience"]
	nodeLeafPlugs = ["friction", "resilience"]
	pass

