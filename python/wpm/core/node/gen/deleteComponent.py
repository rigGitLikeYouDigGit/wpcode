

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
	node : DeleteComponent = None
	pass
class DeleteComponentsPlug(Plug):
	node : DeleteComponent = None
	pass
class EdgeIdMapPlug(Plug):
	node : DeleteComponent = None
	pass
class FaceIdMapPlug(Plug):
	node : DeleteComponent = None
	pass
class InputGeometryPlug(Plug):
	node : DeleteComponent = None
	pass
class OutputGeometryPlug(Plug):
	node : DeleteComponent = None
	pass
class UseOldPolyArchitecturePlug(Plug):
	node : DeleteComponent = None
	pass
class VertexIdMapPlug(Plug):
	node : DeleteComponent = None
	pass
# endregion


# define node class
class DeleteComponent(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	deleteComponents_ : DeleteComponentsPlug = PlugDescriptor("deleteComponents")
	edgeIdMap_ : EdgeIdMapPlug = PlugDescriptor("edgeIdMap")
	faceIdMap_ : FaceIdMapPlug = PlugDescriptor("faceIdMap")
	inputGeometry_ : InputGeometryPlug = PlugDescriptor("inputGeometry")
	outputGeometry_ : OutputGeometryPlug = PlugDescriptor("outputGeometry")
	useOldPolyArchitecture_ : UseOldPolyArchitecturePlug = PlugDescriptor("useOldPolyArchitecture")
	vertexIdMap_ : VertexIdMapPlug = PlugDescriptor("vertexIdMap")

	# node attributes

	typeName = "deleteComponent"
	apiTypeInt = 318
	apiTypeStr = "kDeleteComponent"
	typeIdInt = 1145389908
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "deleteComponents", "edgeIdMap", "faceIdMap", "inputGeometry", "outputGeometry", "useOldPolyArchitecture", "vertexIdMap"]
	nodeLeafPlugs = ["binMembership", "deleteComponents", "edgeIdMap", "faceIdMap", "inputGeometry", "outputGeometry", "useOldPolyArchitecture", "vertexIdMap"]
	pass

