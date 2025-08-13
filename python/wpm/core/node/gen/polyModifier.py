

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyBase = retriever.getNodeCls("PolyBase")
assert PolyBase
if T.TYPE_CHECKING:
	from .. import PolyBase

# add node doc



# region plug type defs
class CacheInputPlug(Plug):
	node : PolyModifier = None
	pass
class EdgeIdMapPlug(Plug):
	node : PolyModifier = None
	pass
class FaceIdMapPlug(Plug):
	node : PolyModifier = None
	pass
class InMeshCachePlug(Plug):
	node : PolyModifier = None
	pass
class InputComponentsPlug(Plug):
	node : PolyModifier = None
	pass
class InputPolymeshPlug(Plug):
	node : PolyModifier = None
	pass
class UseInputCompPlug(Plug):
	node : PolyModifier = None
	pass
class UseOldPolyArchitecturePlug(Plug):
	node : PolyModifier = None
	pass
class VertexIdMapPlug(Plug):
	node : PolyModifier = None
	pass
# endregion


# define node class
class PolyModifier(PolyBase):
	cacheInput_ : CacheInputPlug = PlugDescriptor("cacheInput")
	edgeIdMap_ : EdgeIdMapPlug = PlugDescriptor("edgeIdMap")
	faceIdMap_ : FaceIdMapPlug = PlugDescriptor("faceIdMap")
	inMeshCache_ : InMeshCachePlug = PlugDescriptor("inMeshCache")
	inputComponents_ : InputComponentsPlug = PlugDescriptor("inputComponents")
	inputPolymesh_ : InputPolymeshPlug = PlugDescriptor("inputPolymesh")
	useInputComp_ : UseInputCompPlug = PlugDescriptor("useInputComp")
	useOldPolyArchitecture_ : UseOldPolyArchitecturePlug = PlugDescriptor("useOldPolyArchitecture")
	vertexIdMap_ : VertexIdMapPlug = PlugDescriptor("vertexIdMap")

	# node attributes

	typeName = "polyModifier"
	typeIdInt = 1347243844
	pass

