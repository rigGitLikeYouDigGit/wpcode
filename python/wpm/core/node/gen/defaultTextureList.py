

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : DefaultTextureList = None
	pass
class TexturesPlug(Plug):
	node : DefaultTextureList = None
	pass
# endregion


# define node class
class DefaultTextureList(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	textures_ : TexturesPlug = PlugDescriptor("textures")

	# node attributes

	typeName = "defaultTextureList"
	typeIdInt = 1380209740
	nodeLeafClassAttrs = ["binMembership", "textures"]
	nodeLeafPlugs = ["binMembership", "textures"]
	pass

