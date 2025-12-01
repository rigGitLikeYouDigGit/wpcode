

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
class CyclePlug(Plug):
	node : Instancer = None
	pass
class CycleStepPlug(Plug):
	node : Instancer = None
	pass
class CycleStepUnitPlug(Plug):
	node : Instancer = None
	pass
class DisplayPercentagePlug(Plug):
	node : Instancer = None
	pass
class FillArrayPlug(Plug):
	node : Instancer = None
	pass
class HierarchyCountPlug(Plug):
	node : Instancer = None
	pass
class InputHierarchyPlug(Plug):
	node : Instancer = None
	pass
class InputPointsPlug(Plug):
	node : Instancer = None
	pass
class InstanceCountPlug(Plug):
	node : Instancer = None
	pass
class LevelOfDetailPlug(Plug):
	node : Instancer = None
	pass
class RotationAngleUnitsPlug(Plug):
	node : Instancer = None
	pass
class RotationOrderPlug(Plug):
	node : Instancer = None
	pass
# endregion


# define node class
class Instancer(Transform):
	cycle_ : CyclePlug = PlugDescriptor("cycle")
	cycleStep_ : CycleStepPlug = PlugDescriptor("cycleStep")
	cycleStepUnit_ : CycleStepUnitPlug = PlugDescriptor("cycleStepUnit")
	displayPercentage_ : DisplayPercentagePlug = PlugDescriptor("displayPercentage")
	fillArray_ : FillArrayPlug = PlugDescriptor("fillArray")
	hierarchyCount_ : HierarchyCountPlug = PlugDescriptor("hierarchyCount")
	inputHierarchy_ : InputHierarchyPlug = PlugDescriptor("inputHierarchy")
	inputPoints_ : InputPointsPlug = PlugDescriptor("inputPoints")
	instanceCount_ : InstanceCountPlug = PlugDescriptor("instanceCount")
	levelOfDetail_ : LevelOfDetailPlug = PlugDescriptor("levelOfDetail")
	rotationAngleUnits_ : RotationAngleUnitsPlug = PlugDescriptor("rotationAngleUnits")
	rotationOrder_ : RotationOrderPlug = PlugDescriptor("rotationOrder")

	# node attributes

	typeName = "instancer"
	apiTypeInt = 762
	apiTypeStr = "kInstancer"
	typeIdInt = 1498305364
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["cycle", "cycleStep", "cycleStepUnit", "displayPercentage", "fillArray", "hierarchyCount", "inputHierarchy", "inputPoints", "instanceCount", "levelOfDetail", "rotationAngleUnits", "rotationOrder"]
	nodeLeafPlugs = ["cycle", "cycleStep", "cycleStepUnit", "displayPercentage", "fillArray", "hierarchyCount", "inputHierarchy", "inputPoints", "instanceCount", "levelOfDetail", "rotationAngleUnits", "rotationOrder"]
	pass

