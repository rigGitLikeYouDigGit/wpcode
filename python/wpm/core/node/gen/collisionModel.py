

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	DynBase = Catalogue.DynBase
else:
	from .. import retriever
	DynBase = retriever.getNodeCls("DynBase")
	assert DynBase

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

