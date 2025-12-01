

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
NonExtendedLightShapeNode = retriever.getNodeCls("NonExtendedLightShapeNode")
assert NonExtendedLightShapeNode
if T.TYPE_CHECKING:
	from .. import NonExtendedLightShapeNode

# add node doc



# region plug type defs
class FarPointWorldXPlug(Plug):
	parent : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	node : PointLight = None
	pass
class FarPointWorldYPlug(Plug):
	parent : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	node : PointLight = None
	pass
class FarPointWorldZPlug(Plug):
	parent : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	node : PointLight = None
	pass
class FarPointWorldPlug(Plug):
	farPointWorldX_ : FarPointWorldXPlug = PlugDescriptor("farPointWorldX")
	fwx_ : FarPointWorldXPlug = PlugDescriptor("farPointWorldX")
	farPointWorldY_ : FarPointWorldYPlug = PlugDescriptor("farPointWorldY")
	fwy_ : FarPointWorldYPlug = PlugDescriptor("farPointWorldY")
	farPointWorldZ_ : FarPointWorldZPlug = PlugDescriptor("farPointWorldZ")
	fwz_ : FarPointWorldZPlug = PlugDescriptor("farPointWorldZ")
	node : PointLight = None
	pass
class FogGeometryPlug(Plug):
	node : PointLight = None
	pass
class FogIntensityPlug(Plug):
	node : PointLight = None
	pass
class FogRadiusPlug(Plug):
	node : PointLight = None
	pass
class FogTypePlug(Plug):
	node : PointLight = None
	pass
class LightGlowPlug(Plug):
	node : PointLight = None
	pass
class ObjectTypePlug(Plug):
	node : PointLight = None
	pass
class PointWorldXPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : PointLight = None
	pass
class PointWorldYPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : PointLight = None
	pass
class PointWorldZPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : PointLight = None
	pass
class PointWorldPlug(Plug):
	pointWorldX_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	tx_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pointWorldY_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	ty_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pointWorldZ_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	tz_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	node : PointLight = None
	pass
# endregion


# define node class
class PointLight(NonExtendedLightShapeNode):
	farPointWorldX_ : FarPointWorldXPlug = PlugDescriptor("farPointWorldX")
	farPointWorldY_ : FarPointWorldYPlug = PlugDescriptor("farPointWorldY")
	farPointWorldZ_ : FarPointWorldZPlug = PlugDescriptor("farPointWorldZ")
	farPointWorld_ : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	fogGeometry_ : FogGeometryPlug = PlugDescriptor("fogGeometry")
	fogIntensity_ : FogIntensityPlug = PlugDescriptor("fogIntensity")
	fogRadius_ : FogRadiusPlug = PlugDescriptor("fogRadius")
	fogType_ : FogTypePlug = PlugDescriptor("fogType")
	lightGlow_ : LightGlowPlug = PlugDescriptor("lightGlow")
	objectType_ : ObjectTypePlug = PlugDescriptor("objectType")
	pointWorldX_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pointWorldY_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pointWorldZ_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	pointWorld_ : PointWorldPlug = PlugDescriptor("pointWorld")

	# node attributes

	typeName = "pointLight"
	apiTypeInt = 309
	apiTypeStr = "kPointLight"
	typeIdInt = 1347373396
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["farPointWorldX", "farPointWorldY", "farPointWorldZ", "farPointWorld", "fogGeometry", "fogIntensity", "fogRadius", "fogType", "lightGlow", "objectType", "pointWorldX", "pointWorldY", "pointWorldZ", "pointWorld"]
	nodeLeafPlugs = ["farPointWorld", "fogGeometry", "fogIntensity", "fogRadius", "fogType", "lightGlow", "objectType", "pointWorld"]
	pass

