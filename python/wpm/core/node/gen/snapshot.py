

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
class AnimCurveChangedPlug(Plug):
	node : Snapshot = None
	pass
class BinMembershipPlug(Plug):
	node : Snapshot = None
	pass
class EndTimePlug(Plug):
	node : Snapshot = None
	pass
class FramesPlug(Plug):
	node : Snapshot = None
	pass
class IncrementPlug(Plug):
	node : Snapshot = None
	pass
class InputGeomPlug(Plug):
	node : Snapshot = None
	pass
class InputMatrixPlug(Plug):
	node : Snapshot = None
	pass
class LocalPositionXPlug(Plug):
	parent : LocalPositionPlug = PlugDescriptor("localPosition")
	node : Snapshot = None
	pass
class LocalPositionYPlug(Plug):
	parent : LocalPositionPlug = PlugDescriptor("localPosition")
	node : Snapshot = None
	pass
class LocalPositionZPlug(Plug):
	parent : LocalPositionPlug = PlugDescriptor("localPosition")
	node : Snapshot = None
	pass
class LocalPositionPlug(Plug):
	localPositionX_ : LocalPositionXPlug = PlugDescriptor("localPositionX")
	lpx_ : LocalPositionXPlug = PlugDescriptor("localPositionX")
	localPositionY_ : LocalPositionYPlug = PlugDescriptor("localPositionY")
	lpy_ : LocalPositionYPlug = PlugDescriptor("localPositionY")
	localPositionZ_ : LocalPositionZPlug = PlugDescriptor("localPositionZ")
	lpz_ : LocalPositionZPlug = PlugDescriptor("localPositionZ")
	node : Snapshot = None
	pass
class OutputGeomPlug(Plug):
	node : Snapshot = None
	pass
class PointsPlug(Plug):
	node : Snapshot = None
	pass
class SnapshotObjectPlug(Plug):
	node : Snapshot = None
	pass
class StartTimePlug(Plug):
	node : Snapshot = None
	pass
class UpdatePlug(Plug):
	node : Snapshot = None
	pass
# endregion


# define node class
class Snapshot(_BASE_):
	animCurveChanged_ : AnimCurveChangedPlug = PlugDescriptor("animCurveChanged")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	endTime_ : EndTimePlug = PlugDescriptor("endTime")
	frames_ : FramesPlug = PlugDescriptor("frames")
	increment_ : IncrementPlug = PlugDescriptor("increment")
	inputGeom_ : InputGeomPlug = PlugDescriptor("inputGeom")
	inputMatrix_ : InputMatrixPlug = PlugDescriptor("inputMatrix")
	localPositionX_ : LocalPositionXPlug = PlugDescriptor("localPositionX")
	localPositionY_ : LocalPositionYPlug = PlugDescriptor("localPositionY")
	localPositionZ_ : LocalPositionZPlug = PlugDescriptor("localPositionZ")
	localPosition_ : LocalPositionPlug = PlugDescriptor("localPosition")
	outputGeom_ : OutputGeomPlug = PlugDescriptor("outputGeom")
	points_ : PointsPlug = PlugDescriptor("points")
	snapshotObject_ : SnapshotObjectPlug = PlugDescriptor("snapshotObject")
	startTime_ : StartTimePlug = PlugDescriptor("startTime")
	update_ : UpdatePlug = PlugDescriptor("update")

	# node attributes

	typeName = "snapshot"
	apiTypeInt = 482
	apiTypeStr = "kSnapshot"
	typeIdInt = 1397641300
	MFnCls = om.MFnDagNode
	pass

