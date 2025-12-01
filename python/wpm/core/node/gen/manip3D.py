

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
class ConnectedNodesPlug(Plug):
	node : Manip3D = None
	pass
# endregion


# define node class
class Manip3D(Transform):
	connectedNodes_ : ConnectedNodesPlug = PlugDescriptor("connectedNodes")

	# node attributes

	typeName = "manip3D"
	typeIdInt = 1431131204
	nodeLeafClassAttrs = ["connectedNodes"]
	nodeLeafPlugs = ["connectedNodes"]
	pass

