

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
File = retriever.getNodeCls("File")
assert File
if T.TYPE_CHECKING:
	from .. import File

# add node doc



# region plug type defs
class AlphaPlug(Plug):
	node : PsdFileTex = None
	pass
class AlphaListPlug(Plug):
	node : PsdFileTex = None
	pass
class LayerDepthsPlug(Plug):
	node : PsdFileTex = None
	pass
class LayerIdsPlug(Plug):
	node : PsdFileTex = None
	pass
class LayerSetNamePlug(Plug):
	node : PsdFileTex = None
	pass
class LayerSetsPlug(Plug):
	node : PsdFileTex = None
	pass
# endregion


# define node class
class PsdFileTex(File):
	alpha_ : AlphaPlug = PlugDescriptor("alpha")
	alphaList_ : AlphaListPlug = PlugDescriptor("alphaList")
	layerDepths_ : LayerDepthsPlug = PlugDescriptor("layerDepths")
	layerIds_ : LayerIdsPlug = PlugDescriptor("layerIds")
	layerSetName_ : LayerSetNamePlug = PlugDescriptor("layerSetName")
	layerSets_ : LayerSetsPlug = PlugDescriptor("layerSets")

	# node attributes

	typeName = "psdFileTex"
	typeIdInt = 1347634260
	nodeLeafClassAttrs = ["alpha", "alphaList", "layerDepths", "layerIds", "layerSetName", "layerSets"]
	nodeLeafPlugs = ["alpha", "alphaList", "layerDepths", "layerIds", "layerSetName", "layerSets"]
	pass

