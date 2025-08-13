

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
SubdModifier = retriever.getNodeCls("SubdModifier")
assert SubdModifier
if T.TYPE_CHECKING:
	from .. import SubdModifier

# add node doc



# region plug type defs
class InputMatrixPlug(Plug):
	node : SubdModifierWorld = None
	pass
class ManipMatrixPlug(Plug):
	node : SubdModifierWorld = None
	pass
class WorldSpacePlug(Plug):
	node : SubdModifierWorld = None
	pass
# endregion


# define node class
class SubdModifierWorld(SubdModifier):
	inputMatrix_ : InputMatrixPlug = PlugDescriptor("inputMatrix")
	manipMatrix_ : ManipMatrixPlug = PlugDescriptor("manipMatrix")
	worldSpace_ : WorldSpacePlug = PlugDescriptor("worldSpace")

	# node attributes

	typeName = "subdModifierWorld"
	typeIdInt = 1397572695
	pass

