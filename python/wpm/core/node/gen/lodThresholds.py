

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
class ActiveLevelPlug(Plug):
	node : LodThresholds = None
	pass
class BinMembershipPlug(Plug):
	node : LodThresholds = None
	pass
class CameraXPlug(Plug):
	parent : CameraPlug = PlugDescriptor("camera")
	node : LodThresholds = None
	pass
class CameraYPlug(Plug):
	parent : CameraPlug = PlugDescriptor("camera")
	node : LodThresholds = None
	pass
class CameraZPlug(Plug):
	parent : CameraPlug = PlugDescriptor("camera")
	node : LodThresholds = None
	pass
class CameraPlug(Plug):
	cameraX_ : CameraXPlug = PlugDescriptor("cameraX")
	cax_ : CameraXPlug = PlugDescriptor("cameraX")
	cameraY_ : CameraYPlug = PlugDescriptor("cameraY")
	cay_ : CameraYPlug = PlugDescriptor("cameraY")
	cameraZ_ : CameraZPlug = PlugDescriptor("cameraZ")
	caz_ : CameraZPlug = PlugDescriptor("cameraZ")
	node : LodThresholds = None
	pass
class DistancePlug(Plug):
	node : LodThresholds = None
	pass
class InBoxMaxXPlug(Plug):
	parent : InBoxMaxPlug = PlugDescriptor("inBoxMax")
	node : LodThresholds = None
	pass
class InBoxMaxYPlug(Plug):
	parent : InBoxMaxPlug = PlugDescriptor("inBoxMax")
	node : LodThresholds = None
	pass
class InBoxMaxZPlug(Plug):
	parent : InBoxMaxPlug = PlugDescriptor("inBoxMax")
	node : LodThresholds = None
	pass
class InBoxMaxPlug(Plug):
	inBoxMaxX_ : InBoxMaxXPlug = PlugDescriptor("inBoxMaxX")
	bmax_ : InBoxMaxXPlug = PlugDescriptor("inBoxMaxX")
	inBoxMaxY_ : InBoxMaxYPlug = PlugDescriptor("inBoxMaxY")
	bmay_ : InBoxMaxYPlug = PlugDescriptor("inBoxMaxY")
	inBoxMaxZ_ : InBoxMaxZPlug = PlugDescriptor("inBoxMaxZ")
	bmaz_ : InBoxMaxZPlug = PlugDescriptor("inBoxMaxZ")
	node : LodThresholds = None
	pass
class InBoxMinXPlug(Plug):
	parent : InBoxMinPlug = PlugDescriptor("inBoxMin")
	node : LodThresholds = None
	pass
class InBoxMinYPlug(Plug):
	parent : InBoxMinPlug = PlugDescriptor("inBoxMin")
	node : LodThresholds = None
	pass
class InBoxMinZPlug(Plug):
	parent : InBoxMinPlug = PlugDescriptor("inBoxMin")
	node : LodThresholds = None
	pass
class InBoxMinPlug(Plug):
	inBoxMinX_ : InBoxMinXPlug = PlugDescriptor("inBoxMinX")
	bmix_ : InBoxMinXPlug = PlugDescriptor("inBoxMinX")
	inBoxMinY_ : InBoxMinYPlug = PlugDescriptor("inBoxMinY")
	bmiy_ : InBoxMinYPlug = PlugDescriptor("inBoxMinY")
	inBoxMinZ_ : InBoxMinZPlug = PlugDescriptor("inBoxMinZ")
	bmiz_ : InBoxMinZPlug = PlugDescriptor("inBoxMinZ")
	node : LodThresholds = None
	pass
class OutLevelPlug(Plug):
	node : LodThresholds = None
	pass
class ThresholdPlug(Plug):
	node : LodThresholds = None
	pass
# endregion


# define node class
class LodThresholds(_BASE_):
	activeLevel_ : ActiveLevelPlug = PlugDescriptor("activeLevel")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	cameraX_ : CameraXPlug = PlugDescriptor("cameraX")
	cameraY_ : CameraYPlug = PlugDescriptor("cameraY")
	cameraZ_ : CameraZPlug = PlugDescriptor("cameraZ")
	camera_ : CameraPlug = PlugDescriptor("camera")
	distance_ : DistancePlug = PlugDescriptor("distance")
	inBoxMaxX_ : InBoxMaxXPlug = PlugDescriptor("inBoxMaxX")
	inBoxMaxY_ : InBoxMaxYPlug = PlugDescriptor("inBoxMaxY")
	inBoxMaxZ_ : InBoxMaxZPlug = PlugDescriptor("inBoxMaxZ")
	inBoxMax_ : InBoxMaxPlug = PlugDescriptor("inBoxMax")
	inBoxMinX_ : InBoxMinXPlug = PlugDescriptor("inBoxMinX")
	inBoxMinY_ : InBoxMinYPlug = PlugDescriptor("inBoxMinY")
	inBoxMinZ_ : InBoxMinZPlug = PlugDescriptor("inBoxMinZ")
	inBoxMin_ : InBoxMinPlug = PlugDescriptor("inBoxMin")
	outLevel_ : OutLevelPlug = PlugDescriptor("outLevel")
	threshold_ : ThresholdPlug = PlugDescriptor("threshold")

	# node attributes

	typeName = "lodThresholds"
	apiTypeInt = 771
	apiTypeStr = "kLodThresholds"
	typeIdInt = 1280263252
	MFnCls = om.MFnDependencyNode
	pass

