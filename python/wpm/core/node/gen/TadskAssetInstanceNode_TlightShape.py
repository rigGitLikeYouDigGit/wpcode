

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Light = Catalogue.Light
else:
	from .. import retriever
	Light = retriever.getNodeCls("Light")
	assert Light

# add node doc



# region plug type defs
class AssetIDPlug(Plug):
	node : TadskAssetInstanceNode_TlightShape = None
	pass
class AssetLibraryPlug(Plug):
	node : TadskAssetInstanceNode_TlightShape = None
	pass
# endregion


# define node class
class TadskAssetInstanceNode_TlightShape(Light):
	assetID_ : AssetIDPlug = PlugDescriptor("assetID")
	assetLibrary_ : AssetLibraryPlug = PlugDescriptor("assetLibrary")

	# node attributes

	typeName = "TadskAssetInstanceNode_TlightShape"
	typeIdInt = 1095323219
	nodeLeafClassAttrs = ["assetID", "assetLibrary"]
	nodeLeafPlugs = ["assetID", "assetLibrary"]
	pass

