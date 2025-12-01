

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
_BASE_ = retriever.getNodeCls("_BASE_")
assert _BASE_
if T.TYPE_CHECKING:
	from .. import _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : DefaultShaderList = None
	pass
class ShadersPlug(Plug):
	node : DefaultShaderList = None
	pass
# endregion


# define node class
class DefaultShaderList(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	shaders_ : ShadersPlug = PlugDescriptor("shaders")

	# node attributes

	typeName = "defaultShaderList"
	typeIdInt = 1380209484
	nodeLeafClassAttrs = ["binMembership", "shaders"]
	nodeLeafPlugs = ["binMembership", "shaders"]
	pass

