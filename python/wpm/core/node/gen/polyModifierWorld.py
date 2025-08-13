

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifier = retriever.getNodeCls("PolyModifier")
assert PolyModifier
if T.TYPE_CHECKING:
	from .. import PolyModifier

# add node doc



# region plug type defs
class InputMatrixPlug(Plug):
	node : PolyModifierWorld = None
	pass
class ManipMatrixPlug(Plug):
	node : PolyModifierWorld = None
	pass
class WorldSpacePlug(Plug):
	node : PolyModifierWorld = None
	pass
# endregion


# define node class
class PolyModifierWorld(PolyModifier):
	inputMatrix_ : InputMatrixPlug = PlugDescriptor("inputMatrix")
	manipMatrix_ : ManipMatrixPlug = PlugDescriptor("manipMatrix")
	worldSpace_ : WorldSpacePlug = PlugDescriptor("worldSpace")

	# node attributes

	typeName = "polyModifierWorld"
	typeIdInt = 1347241047
	pass

