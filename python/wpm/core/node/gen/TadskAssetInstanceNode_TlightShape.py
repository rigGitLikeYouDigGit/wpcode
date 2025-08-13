

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Light = retriever.getNodeCls("Light")
assert Light
if T.TYPE_CHECKING:
	from .. import Light

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
	pass

