

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : LightInfo = None
	pass
class LightDirectionXPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : LightInfo = None
	pass
class LightDirectionYPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : LightInfo = None
	pass
class LightDirectionZPlug(Plug):
	parent : LightDirectionPlug = PlugDescriptor("lightDirection")
	node : LightInfo = None
	pass
class LightDirectionPlug(Plug):
	lightDirectionX_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	ldx_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	lightDirectionY_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	ldy_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	lightDirectionZ_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	ldz_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	node : LightInfo = None
	pass
class LightDirectionOnlyPlug(Plug):
	node : LightInfo = None
	pass
class LightPositionXPlug(Plug):
	parent : LightPositionPlug = PlugDescriptor("lightPosition")
	node : LightInfo = None
	pass
class LightPositionYPlug(Plug):
	parent : LightPositionPlug = PlugDescriptor("lightPosition")
	node : LightInfo = None
	pass
class LightPositionZPlug(Plug):
	parent : LightPositionPlug = PlugDescriptor("lightPosition")
	node : LightInfo = None
	pass
class LightPositionPlug(Plug):
	lightPositionX_ : LightPositionXPlug = PlugDescriptor("lightPositionX")
	lpx_ : LightPositionXPlug = PlugDescriptor("lightPositionX")
	lightPositionY_ : LightPositionYPlug = PlugDescriptor("lightPositionY")
	lpy_ : LightPositionYPlug = PlugDescriptor("lightPositionY")
	lightPositionZ_ : LightPositionZPlug = PlugDescriptor("lightPositionZ")
	lpz_ : LightPositionZPlug = PlugDescriptor("lightPositionZ")
	node : LightInfo = None
	pass
class MatrixEyeToWorldPlug(Plug):
	node : LightInfo = None
	pass
class PointCameraXPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : LightInfo = None
	pass
class PointCameraYPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : LightInfo = None
	pass
class PointCameraZPlug(Plug):
	parent : PointCameraPlug = PlugDescriptor("pointCamera")
	node : LightInfo = None
	pass
class PointCameraPlug(Plug):
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	px_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	py_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pz_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	node : LightInfo = None
	pass
class SampleDistancePlug(Plug):
	node : LightInfo = None
	pass
class WorldMatrixPlug(Plug):
	node : LightInfo = None
	pass
# endregion


# define node class
class LightInfo(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	lightDirectionX_ : LightDirectionXPlug = PlugDescriptor("lightDirectionX")
	lightDirectionY_ : LightDirectionYPlug = PlugDescriptor("lightDirectionY")
	lightDirectionZ_ : LightDirectionZPlug = PlugDescriptor("lightDirectionZ")
	lightDirection_ : LightDirectionPlug = PlugDescriptor("lightDirection")
	lightDirectionOnly_ : LightDirectionOnlyPlug = PlugDescriptor("lightDirectionOnly")
	lightPositionX_ : LightPositionXPlug = PlugDescriptor("lightPositionX")
	lightPositionY_ : LightPositionYPlug = PlugDescriptor("lightPositionY")
	lightPositionZ_ : LightPositionZPlug = PlugDescriptor("lightPositionZ")
	lightPosition_ : LightPositionPlug = PlugDescriptor("lightPosition")
	matrixEyeToWorld_ : MatrixEyeToWorldPlug = PlugDescriptor("matrixEyeToWorld")
	pointCameraX_ : PointCameraXPlug = PlugDescriptor("pointCameraX")
	pointCameraY_ : PointCameraYPlug = PlugDescriptor("pointCameraY")
	pointCameraZ_ : PointCameraZPlug = PlugDescriptor("pointCameraZ")
	pointCamera_ : PointCameraPlug = PlugDescriptor("pointCamera")
	sampleDistance_ : SampleDistancePlug = PlugDescriptor("sampleDistance")
	worldMatrix_ : WorldMatrixPlug = PlugDescriptor("worldMatrix")

	# node attributes

	typeName = "lightInfo"
	apiTypeInt = 378
	apiTypeStr = "kLightInfo"
	typeIdInt = 1380731214
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "lightDirectionX", "lightDirectionY", "lightDirectionZ", "lightDirection", "lightDirectionOnly", "lightPositionX", "lightPositionY", "lightPositionZ", "lightPosition", "matrixEyeToWorld", "pointCameraX", "pointCameraY", "pointCameraZ", "pointCamera", "sampleDistance", "worldMatrix"]
	nodeLeafPlugs = ["binMembership", "lightDirection", "lightDirectionOnly", "lightPosition", "matrixEyeToWorld", "pointCamera", "sampleDistance", "worldMatrix"]
	pass

