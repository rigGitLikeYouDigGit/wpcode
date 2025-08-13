

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
RevolvedPrimitive = retriever.getNodeCls("RevolvedPrimitive")
assert RevolvedPrimitive
if T.TYPE_CHECKING:
	from .. import RevolvedPrimitive

# add node doc



# region plug type defs
class UseOldInitBehaviourPlug(Plug):
	node : MakeNurbCone = None
	pass
# endregion


# define node class
class MakeNurbCone(RevolvedPrimitive):
	useOldInitBehaviour_ : UseOldInitBehaviourPlug = PlugDescriptor("useOldInitBehaviour")

	# node attributes

	typeName = "makeNurbCone"
	typeIdInt = 1313033797
	pass

