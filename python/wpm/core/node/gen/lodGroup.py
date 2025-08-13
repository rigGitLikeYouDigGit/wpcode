

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Transform = retriever.getNodeCls("Transform")
assert Transform
if T.TYPE_CHECKING:
	from .. import Transform

# add node doc



# region plug type defs
class ActiveLevelPlug(Plug):
	node : LodGroup = None
	pass
class CameraMatrixPlug(Plug):
	node : LodGroup = None
	pass
class DisplayLevelPlug(Plug):
	node : LodGroup = None
	pass
class DistancePlug(Plug):
	node : LodGroup = None
	pass
class FocalLengthPlug(Plug):
	node : LodGroup = None
	pass
class MaxDistancePlug(Plug):
	node : LodGroup = None
	pass
class MinDistancePlug(Plug):
	node : LodGroup = None
	pass
class MinMaxDistancePlug(Plug):
	node : LodGroup = None
	pass
class OutputPlug(Plug):
	node : LodGroup = None
	pass
class PercentageThresholdPlug(Plug):
	node : LodGroup = None
	pass
class ScreenHeightPercentagePlug(Plug):
	node : LodGroup = None
	pass
class ThresholdPlug(Plug):
	node : LodGroup = None
	pass
class UseScreenHeightPercentagePlug(Plug):
	node : LodGroup = None
	pass
class WorldSpacePlug(Plug):
	node : LodGroup = None
	pass
# endregion


# define node class
class LodGroup(Transform):
	activeLevel_ : ActiveLevelPlug = PlugDescriptor("activeLevel")
	cameraMatrix_ : CameraMatrixPlug = PlugDescriptor("cameraMatrix")
	displayLevel_ : DisplayLevelPlug = PlugDescriptor("displayLevel")
	distance_ : DistancePlug = PlugDescriptor("distance")
	focalLength_ : FocalLengthPlug = PlugDescriptor("focalLength")
	maxDistance_ : MaxDistancePlug = PlugDescriptor("maxDistance")
	minDistance_ : MinDistancePlug = PlugDescriptor("minDistance")
	minMaxDistance_ : MinMaxDistancePlug = PlugDescriptor("minMaxDistance")
	output_ : OutputPlug = PlugDescriptor("output")
	percentageThreshold_ : PercentageThresholdPlug = PlugDescriptor("percentageThreshold")
	screenHeightPercentage_ : ScreenHeightPercentagePlug = PlugDescriptor("screenHeightPercentage")
	threshold_ : ThresholdPlug = PlugDescriptor("threshold")
	useScreenHeightPercentage_ : UseScreenHeightPercentagePlug = PlugDescriptor("useScreenHeightPercentage")
	worldSpace_ : WorldSpacePlug = PlugDescriptor("worldSpace")

	# node attributes

	typeName = "lodGroup"
	apiTypeInt = 773
	apiTypeStr = "kLodGroup"
	typeIdInt = 1280263239
	MFnCls = om.MFnTransform
	pass

