

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Shape = Catalogue.Shape
else:
	from .. import retriever
	Shape = retriever.getNodeCls("Shape")
	assert Shape

# add node doc



# region plug type defs
class AuxiliariesOwnedPlug(Plug):
	node : DynHolder = None
	pass
class ConnectionsToMePlug(Plug):
	node : DynHolder = None
	pass
# endregion


# define node class
class DynHolder(Shape):
	auxiliariesOwned_ : AuxiliariesOwnedPlug = PlugDescriptor("auxiliariesOwned")
	connectionsToMe_ : ConnectionsToMePlug = PlugDescriptor("connectionsToMe")

	# node attributes

	typeName = "dynHolder"
	typeIdInt = 1497910340
	nodeLeafClassAttrs = ["auxiliariesOwned", "connectionsToMe"]
	nodeLeafPlugs = ["auxiliariesOwned", "connectionsToMe"]
	pass

