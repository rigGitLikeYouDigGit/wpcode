

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
class XCoordinatePlug(Plug):
	parent : AllCoordinatesPlug = PlugDescriptor("allCoordinates")
	node : MotionPath = None
	pass
class YCoordinatePlug(Plug):
	parent : AllCoordinatesPlug = PlugDescriptor("allCoordinates")
	node : MotionPath = None
	pass
class ZCoordinatePlug(Plug):
	parent : AllCoordinatesPlug = PlugDescriptor("allCoordinates")
	node : MotionPath = None
	pass
class AllCoordinatesPlug(Plug):
	xCoordinate_ : XCoordinatePlug = PlugDescriptor("xCoordinate")
	xc_ : XCoordinatePlug = PlugDescriptor("xCoordinate")
	yCoordinate_ : YCoordinatePlug = PlugDescriptor("yCoordinate")
	yc_ : YCoordinatePlug = PlugDescriptor("yCoordinate")
	zCoordinate_ : ZCoordinatePlug = PlugDescriptor("zCoordinate")
	zc_ : ZCoordinatePlug = PlugDescriptor("zCoordinate")
	node : MotionPath = None
	pass
class BankPlug(Plug):
	node : MotionPath = None
	pass
class BankLimitPlug(Plug):
	node : MotionPath = None
	pass
class BankScalePlug(Plug):
	node : MotionPath = None
	pass
class BinMembershipPlug(Plug):
	node : MotionPath = None
	pass
class FlowNodePlug(Plug):
	node : MotionPath = None
	pass
class FollowPlug(Plug):
	node : MotionPath = None
	pass
class FractionModePlug(Plug):
	node : MotionPath = None
	pass
class FrontAxisPlug(Plug):
	node : MotionPath = None
	pass
class FrontTwistPlug(Plug):
	node : MotionPath = None
	pass
class GeometryPathPlug(Plug):
	node : MotionPath = None
	pass
class InverseFrontPlug(Plug):
	node : MotionPath = None
	pass
class InverseUpPlug(Plug):
	node : MotionPath = None
	pass
class NormalPlug(Plug):
	node : MotionPath = None
	pass
class OrientMatrixPlug(Plug):
	node : MotionPath = None
	pass
class OrientationMarkerTimePlug(Plug):
	node : MotionPath = None
	pass
class PositionMarkerTimePlug(Plug):
	node : MotionPath = None
	pass
class RotateXPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : MotionPath = None
	pass
class RotateYPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : MotionPath = None
	pass
class RotateZPlug(Plug):
	parent : RotatePlug = PlugDescriptor("rotate")
	node : MotionPath = None
	pass
class RotatePlug(Plug):
	rotateX_ : RotateXPlug = PlugDescriptor("rotateX")
	rx_ : RotateXPlug = PlugDescriptor("rotateX")
	rotateY_ : RotateYPlug = PlugDescriptor("rotateY")
	ry_ : RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_ : RotateZPlug = PlugDescriptor("rotateZ")
	rz_ : RotateZPlug = PlugDescriptor("rotateZ")
	node : MotionPath = None
	pass
class RotateOrderPlug(Plug):
	node : MotionPath = None
	pass
class SideTwistPlug(Plug):
	node : MotionPath = None
	pass
class UValuePlug(Plug):
	node : MotionPath = None
	pass
class UpAxisPlug(Plug):
	node : MotionPath = None
	pass
class UpTwistPlug(Plug):
	node : MotionPath = None
	pass
class UpdateOMPlug(Plug):
	node : MotionPath = None
	pass
class WorldUpMatrixPlug(Plug):
	node : MotionPath = None
	pass
class WorldUpTypePlug(Plug):
	node : MotionPath = None
	pass
class WorldUpVectorXPlug(Plug):
	parent : WorldUpVectorPlug = PlugDescriptor("worldUpVector")
	node : MotionPath = None
	pass
class WorldUpVectorYPlug(Plug):
	parent : WorldUpVectorPlug = PlugDescriptor("worldUpVector")
	node : MotionPath = None
	pass
class WorldUpVectorZPlug(Plug):
	parent : WorldUpVectorPlug = PlugDescriptor("worldUpVector")
	node : MotionPath = None
	pass
class WorldUpVectorPlug(Plug):
	worldUpVectorX_ : WorldUpVectorXPlug = PlugDescriptor("worldUpVectorX")
	wux_ : WorldUpVectorXPlug = PlugDescriptor("worldUpVectorX")
	worldUpVectorY_ : WorldUpVectorYPlug = PlugDescriptor("worldUpVectorY")
	wuy_ : WorldUpVectorYPlug = PlugDescriptor("worldUpVectorY")
	worldUpVectorZ_ : WorldUpVectorZPlug = PlugDescriptor("worldUpVectorZ")
	wuz_ : WorldUpVectorZPlug = PlugDescriptor("worldUpVectorZ")
	node : MotionPath = None
	pass
# endregion


# define node class
class MotionPath(_BASE_):
	xCoordinate_ : XCoordinatePlug = PlugDescriptor("xCoordinate")
	yCoordinate_ : YCoordinatePlug = PlugDescriptor("yCoordinate")
	zCoordinate_ : ZCoordinatePlug = PlugDescriptor("zCoordinate")
	allCoordinates_ : AllCoordinatesPlug = PlugDescriptor("allCoordinates")
	bank_ : BankPlug = PlugDescriptor("bank")
	bankLimit_ : BankLimitPlug = PlugDescriptor("bankLimit")
	bankScale_ : BankScalePlug = PlugDescriptor("bankScale")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	flowNode_ : FlowNodePlug = PlugDescriptor("flowNode")
	follow_ : FollowPlug = PlugDescriptor("follow")
	fractionMode_ : FractionModePlug = PlugDescriptor("fractionMode")
	frontAxis_ : FrontAxisPlug = PlugDescriptor("frontAxis")
	frontTwist_ : FrontTwistPlug = PlugDescriptor("frontTwist")
	geometryPath_ : GeometryPathPlug = PlugDescriptor("geometryPath")
	inverseFront_ : InverseFrontPlug = PlugDescriptor("inverseFront")
	inverseUp_ : InverseUpPlug = PlugDescriptor("inverseUp")
	normal_ : NormalPlug = PlugDescriptor("normal")
	orientMatrix_ : OrientMatrixPlug = PlugDescriptor("orientMatrix")
	orientationMarkerTime_ : OrientationMarkerTimePlug = PlugDescriptor("orientationMarkerTime")
	positionMarkerTime_ : PositionMarkerTimePlug = PlugDescriptor("positionMarkerTime")
	rotateX_ : RotateXPlug = PlugDescriptor("rotateX")
	rotateY_ : RotateYPlug = PlugDescriptor("rotateY")
	rotateZ_ : RotateZPlug = PlugDescriptor("rotateZ")
	rotate_ : RotatePlug = PlugDescriptor("rotate")
	rotateOrder_ : RotateOrderPlug = PlugDescriptor("rotateOrder")
	sideTwist_ : SideTwistPlug = PlugDescriptor("sideTwist")
	uValue_ : UValuePlug = PlugDescriptor("uValue")
	upAxis_ : UpAxisPlug = PlugDescriptor("upAxis")
	upTwist_ : UpTwistPlug = PlugDescriptor("upTwist")
	updateOM_ : UpdateOMPlug = PlugDescriptor("updateOM")
	worldUpMatrix_ : WorldUpMatrixPlug = PlugDescriptor("worldUpMatrix")
	worldUpType_ : WorldUpTypePlug = PlugDescriptor("worldUpType")
	worldUpVectorX_ : WorldUpVectorXPlug = PlugDescriptor("worldUpVectorX")
	worldUpVectorY_ : WorldUpVectorYPlug = PlugDescriptor("worldUpVectorY")
	worldUpVectorZ_ : WorldUpVectorZPlug = PlugDescriptor("worldUpVectorZ")
	worldUpVector_ : WorldUpVectorPlug = PlugDescriptor("worldUpVector")

	# node attributes

	typeName = "motionPath"
	apiTypeInt = 445
	apiTypeStr = "kMotionPath"
	typeIdInt = 1297110088
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["xCoordinate", "yCoordinate", "zCoordinate", "allCoordinates", "bank", "bankLimit", "bankScale", "binMembership", "flowNode", "follow", "fractionMode", "frontAxis", "frontTwist", "geometryPath", "inverseFront", "inverseUp", "normal", "orientMatrix", "orientationMarkerTime", "positionMarkerTime", "rotateX", "rotateY", "rotateZ", "rotate", "rotateOrder", "sideTwist", "uValue", "upAxis", "upTwist", "updateOM", "worldUpMatrix", "worldUpType", "worldUpVectorX", "worldUpVectorY", "worldUpVectorZ", "worldUpVector"]
	nodeLeafPlugs = ["allCoordinates", "bank", "bankLimit", "bankScale", "binMembership", "flowNode", "follow", "fractionMode", "frontAxis", "frontTwist", "geometryPath", "inverseFront", "inverseUp", "normal", "orientMatrix", "orientationMarkerTime", "positionMarkerTime", "rotate", "rotateOrder", "sideTwist", "uValue", "upAxis", "upTwist", "updateOM", "worldUpMatrix", "worldUpType", "worldUpVector"]
	pass

