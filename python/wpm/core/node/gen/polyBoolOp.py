

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyUnite = retriever.getNodeCls("PolyUnite")
assert PolyUnite
if T.TYPE_CHECKING:
	from .. import PolyUnite

# add node doc



# region plug type defs
class ClassificationPlug(Plug):
	node : PolyBoolOp = None
	pass
class FaceAreaThresholdPlug(Plug):
	node : PolyBoolOp = None
	pass
class OperationPlug(Plug):
	node : PolyBoolOp = None
	pass
class PreserveColorPlug(Plug):
	node : PolyBoolOp = None
	pass
class UseThresholdsPlug(Plug):
	node : PolyBoolOp = None
	pass
class VertexDistanceThresholdPlug(Plug):
	node : PolyBoolOp = None
	pass
# endregion


# define node class
class PolyBoolOp(PolyUnite):
	classification_ : ClassificationPlug = PlugDescriptor("classification")
	faceAreaThreshold_ : FaceAreaThresholdPlug = PlugDescriptor("faceAreaThreshold")
	operation_ : OperationPlug = PlugDescriptor("operation")
	preserveColor_ : PreserveColorPlug = PlugDescriptor("preserveColor")
	useThresholds_ : UseThresholdsPlug = PlugDescriptor("useThresholds")
	vertexDistanceThreshold_ : VertexDistanceThresholdPlug = PlugDescriptor("vertexDistanceThreshold")

	# node attributes

	typeName = "polyBoolOp"
	apiTypeInt = 617
	apiTypeStr = "kPolyBoolOp"
	typeIdInt = 1346522960
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["classification", "faceAreaThreshold", "operation", "preserveColor", "useThresholds", "vertexDistanceThreshold"]
	nodeLeafPlugs = ["classification", "faceAreaThreshold", "operation", "preserveColor", "useThresholds", "vertexDistanceThreshold"]
	pass

