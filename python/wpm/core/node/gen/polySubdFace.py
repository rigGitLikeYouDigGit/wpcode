

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
class DivisionsPlug(Plug):
	node : PolySubdFace = None
	pass
class DivisionsUPlug(Plug):
	node : PolySubdFace = None
	pass
class DivisionsVPlug(Plug):
	node : PolySubdFace = None
	pass
class ModePlug(Plug):
	node : PolySubdFace = None
	pass
class SubdMethodPlug(Plug):
	node : PolySubdFace = None
	pass
# endregion


# define node class
class PolySubdFace(PolyModifier):
	divisions_ : DivisionsPlug = PlugDescriptor("divisions")
	divisionsU_ : DivisionsUPlug = PlugDescriptor("divisionsU")
	divisionsV_ : DivisionsVPlug = PlugDescriptor("divisionsV")
	mode_ : ModePlug = PlugDescriptor("mode")
	subdMethod_ : SubdMethodPlug = PlugDescriptor("subdMethod")

	# node attributes

	typeName = "polySubdFace"
	typeIdInt = 1347638598
	nodeLeafClassAttrs = ["divisions", "divisionsU", "divisionsV", "mode", "subdMethod"]
	nodeLeafPlugs = ["divisions", "divisionsU", "divisionsV", "mode", "subdMethod"]
	pass

