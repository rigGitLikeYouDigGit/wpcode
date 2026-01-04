

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	NonExtendedLightShapeNode = Catalogue.NonExtendedLightShapeNode
else:
	from .. import retriever
	NonExtendedLightShapeNode = retriever.getNodeCls("NonExtendedLightShapeNode")
	assert NonExtendedLightShapeNode

# add node doc



# region plug type defs
class LightAnglePlug(Plug):
	node : DirectionalLight = None
	pass
class ObjectTypePlug(Plug):
	node : DirectionalLight = None
	pass
class PointWorldXPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : DirectionalLight = None
	pass
class PointWorldYPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : DirectionalLight = None
	pass
class PointWorldZPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : DirectionalLight = None
	pass
class PointWorldPlug(Plug):
	pointWorldX_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	tx_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pointWorldY_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	ty_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pointWorldZ_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	tz_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	node : DirectionalLight = None
	pass
class UseLightPositionPlug(Plug):
	node : DirectionalLight = None
	pass
# endregion


# define node class
class DirectionalLight(NonExtendedLightShapeNode):
	lightAngle_ : LightAnglePlug = PlugDescriptor("lightAngle")
	objectType_ : ObjectTypePlug = PlugDescriptor("objectType")
	pointWorldX_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pointWorldY_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pointWorldZ_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	pointWorld_ : PointWorldPlug = PlugDescriptor("pointWorld")
	useLightPosition_ : UseLightPositionPlug = PlugDescriptor("useLightPosition")

	# node attributes

	typeName = "directionalLight"
	apiTypeInt = 308
	apiTypeStr = "kDirectionalLight"
	typeIdInt = 1145655884
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["lightAngle", "objectType", "pointWorldX", "pointWorldY", "pointWorldZ", "pointWorld", "useLightPosition"]
	nodeLeafPlugs = ["lightAngle", "objectType", "pointWorld", "useLightPosition"]
	pass

