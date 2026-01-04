

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
class LightGlowPlug(Plug):
	node : AreaLight = None
	pass
class NormalCameraXPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : AreaLight = None
	pass
class NormalCameraYPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : AreaLight = None
	pass
class NormalCameraZPlug(Plug):
	parent : NormalCameraPlug = PlugDescriptor("normalCamera")
	node : AreaLight = None
	pass
class NormalCameraPlug(Plug):
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	nx_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	ny_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	nz_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	node : AreaLight = None
	pass
class NormalizePlug(Plug):
	node : AreaLight = None
	pass
class ObjectTypePlug(Plug):
	node : AreaLight = None
	pass
class PointWorldXPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : AreaLight = None
	pass
class PointWorldYPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : AreaLight = None
	pass
class PointWorldZPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : AreaLight = None
	pass
class PointWorldPlug(Plug):
	pointWorldX_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	tx_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pointWorldY_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	ty_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pointWorldZ_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	tz_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	node : AreaLight = None
	pass
# endregion


# define node class
class AreaLight(NonExtendedLightShapeNode):
	lightGlow_ : LightGlowPlug = PlugDescriptor("lightGlow")
	normalCameraX_ : NormalCameraXPlug = PlugDescriptor("normalCameraX")
	normalCameraY_ : NormalCameraYPlug = PlugDescriptor("normalCameraY")
	normalCameraZ_ : NormalCameraZPlug = PlugDescriptor("normalCameraZ")
	normalCamera_ : NormalCameraPlug = PlugDescriptor("normalCamera")
	normalize_ : NormalizePlug = PlugDescriptor("normalize")
	objectType_ : ObjectTypePlug = PlugDescriptor("objectType")
	pointWorldX_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pointWorldY_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pointWorldZ_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	pointWorld_ : PointWorldPlug = PlugDescriptor("pointWorld")

	# node attributes

	typeName = "areaLight"
	apiTypeInt = 305
	apiTypeStr = "kAreaLight"
	typeIdInt = 1095912532
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["lightGlow", "normalCameraX", "normalCameraY", "normalCameraZ", "normalCamera", "normalize", "objectType", "pointWorldX", "pointWorldY", "pointWorldZ", "pointWorld"]
	nodeLeafPlugs = ["lightGlow", "normalCamera", "normalize", "objectType", "pointWorld"]
	pass

