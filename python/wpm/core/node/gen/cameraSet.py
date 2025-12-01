

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
	node : CameraSet = None
	pass
class ActivePlug(Plug):
	parent : CameraLayerPlug = PlugDescriptor("cameraLayer")
	node : CameraSet = None
	pass
class CameraPlug(Plug):
	parent : CameraLayerPlug = PlugDescriptor("cameraLayer")
	node : CameraSet = None
	pass
class ClearDepthPlug(Plug):
	parent : CameraLayerPlug = PlugDescriptor("cameraLayer")
	node : CameraSet = None
	pass
class OrderPlug(Plug):
	parent : CameraLayerPlug = PlugDescriptor("cameraLayer")
	node : CameraSet = None
	pass
class SceneDataPlug(Plug):
	parent : CameraLayerPlug = PlugDescriptor("cameraLayer")
	node : CameraSet = None
	pass
class CameraLayerPlug(Plug):
	active_ : ActivePlug = PlugDescriptor("active")
	act_ : ActivePlug = PlugDescriptor("active")
	camera_ : CameraPlug = PlugDescriptor("camera")
	cam_ : CameraPlug = PlugDescriptor("camera")
	clearDepth_ : ClearDepthPlug = PlugDescriptor("clearDepth")
	cld_ : ClearDepthPlug = PlugDescriptor("clearDepth")
	order_ : OrderPlug = PlugDescriptor("order")
	ord_ : OrderPlug = PlugDescriptor("order")
	sceneData_ : SceneDataPlug = PlugDescriptor("sceneData")
	sda_ : SceneDataPlug = PlugDescriptor("sceneData")
	node : CameraSet = None
	pass
# endregion


# define node class
class CameraSet(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	active_ : ActivePlug = PlugDescriptor("active")
	camera_ : CameraPlug = PlugDescriptor("camera")
	clearDepth_ : ClearDepthPlug = PlugDescriptor("clearDepth")
	order_ : OrderPlug = PlugDescriptor("order")
	sceneData_ : SceneDataPlug = PlugDescriptor("sceneData")
	cameraLayer_ : CameraLayerPlug = PlugDescriptor("cameraLayer")

	# node attributes

	typeName = "cameraSet"
	apiTypeInt = 1011
	apiTypeStr = "kCameraSet"
	typeIdInt = 1146246226
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "active", "camera", "clearDepth", "order", "sceneData", "cameraLayer"]
	nodeLeafPlugs = ["binMembership", "cameraLayer"]
	pass

