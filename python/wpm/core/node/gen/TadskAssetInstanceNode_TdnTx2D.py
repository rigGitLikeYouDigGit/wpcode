

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Texture2d = retriever.getNodeCls("Texture2d")
assert Texture2d
if T.TYPE_CHECKING:
	from .. import Texture2d

# add node doc



# region plug type defs
class AssetIDPlug(Plug):
	node : TadskAssetInstanceNode_TdnTx2D = None
	pass
class AssetLibraryPlug(Plug):
	node : TadskAssetInstanceNode_TdnTx2D = None
	pass
# endregion


# define node class
class TadskAssetInstanceNode_TdnTx2D(Texture2d):
	assetID_ : AssetIDPlug = PlugDescriptor("assetID")
	assetLibrary_ : AssetLibraryPlug = PlugDescriptor("assetLibrary")

	# node attributes

	typeName = "TadskAssetInstanceNode_TdnTx2D"
	typeIdInt = 1095323220
	nodeLeafClassAttrs = ["assetID", "assetLibrary"]
	nodeLeafPlugs = ["assetID", "assetLibrary"]
	pass

