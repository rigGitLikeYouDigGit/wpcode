

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Collection = retriever.getNodeCls("Collection")
assert Collection
if T.TYPE_CHECKING:
	from .. import Collection

# add node doc



# region plug type defs
class NumIsolatedRenderSettingsChildrenPlug(Plug):
	node : RenderSettingsCollection = None
	pass
# endregion


# define node class
class RenderSettingsCollection(Collection):
	numIsolatedRenderSettingsChildren_ : NumIsolatedRenderSettingsChildrenPlug = PlugDescriptor("numIsolatedRenderSettingsChildren")

	# node attributes

	typeName = "renderSettingsCollection"
	typeIdInt = 1476395925
	nodeLeafClassAttrs = ["numIsolatedRenderSettingsChildren"]
	nodeLeafPlugs = ["numIsolatedRenderSettingsChildren"]
	pass

