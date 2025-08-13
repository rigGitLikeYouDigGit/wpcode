

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifierWorld = retriever.getNodeCls("PolyModifierWorld")
assert PolyModifierWorld
if T.TYPE_CHECKING:
	from .. import PolyModifierWorld

# add node doc



# region plug type defs
class CreasePlug(Plug):
	node : PolyCrease = None
	pass
class CreaseVertexPlug(Plug):
	node : PolyCrease = None
	pass
class InputVertexComponentsPlug(Plug):
	node : PolyCrease = None
	pass
class OperationPlug(Plug):
	node : PolyCrease = None
	pass
# endregion


# define node class
class PolyCrease(PolyModifierWorld):
	crease_ : CreasePlug = PlugDescriptor("crease")
	creaseVertex_ : CreaseVertexPlug = PlugDescriptor("creaseVertex")
	inputVertexComponents_ : InputVertexComponentsPlug = PlugDescriptor("inputVertexComponents")
	operation_ : OperationPlug = PlugDescriptor("operation")

	# node attributes

	typeName = "polyCrease"
	typeIdInt = 1346589267
	pass

