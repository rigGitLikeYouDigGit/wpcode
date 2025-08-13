

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
class AssetIDPlug(Plug):
	node : TadskAssetInstanceNode_TdependNode = None
	pass
class AssetLibraryPlug(Plug):
	node : TadskAssetInstanceNode_TdependNode = None
	pass
class BinMembershipPlug(Plug):
	node : TadskAssetInstanceNode_TdependNode = None
	pass
# endregion


# define node class
class TadskAssetInstanceNode_TdependNode(_BASE_):
	assetID_ : AssetIDPlug = PlugDescriptor("assetID")
	assetLibrary_ : AssetLibraryPlug = PlugDescriptor("assetLibrary")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")

	# node attributes

	typeName = "TadskAssetInstanceNode_TdependNode"
	typeIdInt = 1095323204
	pass

