

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
class BarnDoorsPlug(Plug):
	node : SpotLight = None
	pass
class BottomBarnDoorPlug(Plug):
	node : SpotLight = None
	pass
class ConeAnglePlug(Plug):
	node : SpotLight = None
	pass
class DropoffPlug(Plug):
	node : SpotLight = None
	pass
class EndDistance1Plug(Plug):
	node : SpotLight = None
	pass
class EndDistance2Plug(Plug):
	node : SpotLight = None
	pass
class EndDistance3Plug(Plug):
	node : SpotLight = None
	pass
class FarPointWorldXPlug(Plug):
	parent : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	node : SpotLight = None
	pass
class FarPointWorldYPlug(Plug):
	parent : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	node : SpotLight = None
	pass
class FarPointWorldZPlug(Plug):
	parent : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	node : SpotLight = None
	pass
class FarPointWorldPlug(Plug):
	farPointWorldX_ : FarPointWorldXPlug = PlugDescriptor("farPointWorldX")
	fx_ : FarPointWorldXPlug = PlugDescriptor("farPointWorldX")
	farPointWorldY_ : FarPointWorldYPlug = PlugDescriptor("farPointWorldY")
	fy_ : FarPointWorldYPlug = PlugDescriptor("farPointWorldY")
	farPointWorldZ_ : FarPointWorldZPlug = PlugDescriptor("farPointWorldZ")
	fz_ : FarPointWorldZPlug = PlugDescriptor("farPointWorldZ")
	node : SpotLight = None
	pass
class FogGeometryPlug(Plug):
	node : SpotLight = None
	pass
class FogIntensityPlug(Plug):
	node : SpotLight = None
	pass
class FogSpreadPlug(Plug):
	node : SpotLight = None
	pass
class LeftBarnDoorPlug(Plug):
	node : SpotLight = None
	pass
class LightGlowPlug(Plug):
	node : SpotLight = None
	pass
class ObjectTypePlug(Plug):
	node : SpotLight = None
	pass
class PenumbraAnglePlug(Plug):
	node : SpotLight = None
	pass
class PointWorldXPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : SpotLight = None
	pass
class PointWorldYPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : SpotLight = None
	pass
class PointWorldZPlug(Plug):
	parent : PointWorldPlug = PlugDescriptor("pointWorld")
	node : SpotLight = None
	pass
class PointWorldPlug(Plug):
	pointWorldX_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	tx_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pointWorldY_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	ty_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pointWorldZ_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	tz_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	node : SpotLight = None
	pass
class PsIllumSamplesPlug(Plug):
	node : SpotLight = None
	pass
class RayDirectionXPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : SpotLight = None
	pass
class RayDirectionYPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : SpotLight = None
	pass
class RayDirectionZPlug(Plug):
	parent : RayDirectionPlug = PlugDescriptor("rayDirection")
	node : SpotLight = None
	pass
class RayDirectionPlug(Plug):
	rayDirectionX_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rdx_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rayDirectionY_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rdy_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rayDirectionZ_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	rdz_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	node : SpotLight = None
	pass
class RightBarnDoorPlug(Plug):
	node : SpotLight = None
	pass
class StartDistance1Plug(Plug):
	node : SpotLight = None
	pass
class StartDistance2Plug(Plug):
	node : SpotLight = None
	pass
class StartDistance3Plug(Plug):
	node : SpotLight = None
	pass
class TopBarnDoorPlug(Plug):
	node : SpotLight = None
	pass
class UseDecayRegionsPlug(Plug):
	node : SpotLight = None
	pass
# endregion


# define node class
class SpotLight(NonExtendedLightShapeNode):
	barnDoors_ : BarnDoorsPlug = PlugDescriptor("barnDoors")
	bottomBarnDoor_ : BottomBarnDoorPlug = PlugDescriptor("bottomBarnDoor")
	coneAngle_ : ConeAnglePlug = PlugDescriptor("coneAngle")
	dropoff_ : DropoffPlug = PlugDescriptor("dropoff")
	endDistance1_ : EndDistance1Plug = PlugDescriptor("endDistance1")
	endDistance2_ : EndDistance2Plug = PlugDescriptor("endDistance2")
	endDistance3_ : EndDistance3Plug = PlugDescriptor("endDistance3")
	farPointWorldX_ : FarPointWorldXPlug = PlugDescriptor("farPointWorldX")
	farPointWorldY_ : FarPointWorldYPlug = PlugDescriptor("farPointWorldY")
	farPointWorldZ_ : FarPointWorldZPlug = PlugDescriptor("farPointWorldZ")
	farPointWorld_ : FarPointWorldPlug = PlugDescriptor("farPointWorld")
	fogGeometry_ : FogGeometryPlug = PlugDescriptor("fogGeometry")
	fogIntensity_ : FogIntensityPlug = PlugDescriptor("fogIntensity")
	fogSpread_ : FogSpreadPlug = PlugDescriptor("fogSpread")
	leftBarnDoor_ : LeftBarnDoorPlug = PlugDescriptor("leftBarnDoor")
	lightGlow_ : LightGlowPlug = PlugDescriptor("lightGlow")
	objectType_ : ObjectTypePlug = PlugDescriptor("objectType")
	penumbraAngle_ : PenumbraAnglePlug = PlugDescriptor("penumbraAngle")
	pointWorldX_ : PointWorldXPlug = PlugDescriptor("pointWorldX")
	pointWorldY_ : PointWorldYPlug = PlugDescriptor("pointWorldY")
	pointWorldZ_ : PointWorldZPlug = PlugDescriptor("pointWorldZ")
	pointWorld_ : PointWorldPlug = PlugDescriptor("pointWorld")
	psIllumSamples_ : PsIllumSamplesPlug = PlugDescriptor("psIllumSamples")
	rayDirectionX_ : RayDirectionXPlug = PlugDescriptor("rayDirectionX")
	rayDirectionY_ : RayDirectionYPlug = PlugDescriptor("rayDirectionY")
	rayDirectionZ_ : RayDirectionZPlug = PlugDescriptor("rayDirectionZ")
	rayDirection_ : RayDirectionPlug = PlugDescriptor("rayDirection")
	rightBarnDoor_ : RightBarnDoorPlug = PlugDescriptor("rightBarnDoor")
	startDistance1_ : StartDistance1Plug = PlugDescriptor("startDistance1")
	startDistance2_ : StartDistance2Plug = PlugDescriptor("startDistance2")
	startDistance3_ : StartDistance3Plug = PlugDescriptor("startDistance3")
	topBarnDoor_ : TopBarnDoorPlug = PlugDescriptor("topBarnDoor")
	useDecayRegions_ : UseDecayRegionsPlug = PlugDescriptor("useDecayRegions")

	# node attributes

	typeName = "spotLight"
	apiTypeInt = 310
	apiTypeStr = "kSpotLight"
	typeIdInt = 1397773388
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["barnDoors", "bottomBarnDoor", "coneAngle", "dropoff", "endDistance1", "endDistance2", "endDistance3", "farPointWorldX", "farPointWorldY", "farPointWorldZ", "farPointWorld", "fogGeometry", "fogIntensity", "fogSpread", "leftBarnDoor", "lightGlow", "objectType", "penumbraAngle", "pointWorldX", "pointWorldY", "pointWorldZ", "pointWorld", "psIllumSamples", "rayDirectionX", "rayDirectionY", "rayDirectionZ", "rayDirection", "rightBarnDoor", "startDistance1", "startDistance2", "startDistance3", "topBarnDoor", "useDecayRegions"]
	nodeLeafPlugs = ["barnDoors", "bottomBarnDoor", "coneAngle", "dropoff", "endDistance1", "endDistance2", "endDistance3", "farPointWorld", "fogGeometry", "fogIntensity", "fogSpread", "leftBarnDoor", "lightGlow", "objectType", "penumbraAngle", "pointWorld", "psIllumSamples", "rayDirection", "rightBarnDoor", "startDistance1", "startDistance2", "startDistance3", "topBarnDoor", "useDecayRegions"]
	pass

