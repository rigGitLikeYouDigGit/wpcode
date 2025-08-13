

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifierWorld = retriever.getNodeCls("PolyModifierWorld")
assert PolyModifierWorld
if T.TYPE_CHECKING:
	from .. import PolyModifierWorld

# add node doc



# region plug type defs
class AxisPlug(Plug):
	node : PolyMirror = None
	pass
class AxisDirectionPlug(Plug):
	node : PolyMirror = None
	pass
class CompIdPlug(Plug):
	node : PolyMirror = None
	pass
class CutMeshPlug(Plug):
	node : PolyMirror = None
	pass
class DirectionPlug(Plug):
	node : PolyMirror = None
	pass
class FirstNewFacePlug(Plug):
	node : PolyMirror = None
	pass
class FlipUVsPlug(Plug):
	node : PolyMirror = None
	pass
class KeepVertexIDsPlug(Plug):
	node : PolyMirror = None
	pass
class LastNewFacePlug(Plug):
	node : PolyMirror = None
	pass
class Maya2017Plug(Plug):
	node : PolyMirror = None
	pass
class MergeModePlug(Plug):
	node : PolyMirror = None
	pass
class MergeThresholdPlug(Plug):
	node : PolyMirror = None
	pass
class MergeThresholdTypePlug(Plug):
	node : PolyMirror = None
	pass
class MirrorAxisPlug(Plug):
	node : PolyMirror = None
	pass
class MirrorPlaneCenterXPlug(Plug):
	parent : MirrorPlaneCenterPlug = PlugDescriptor("mirrorPlaneCenter")
	node : PolyMirror = None
	pass
class MirrorPlaneCenterYPlug(Plug):
	parent : MirrorPlaneCenterPlug = PlugDescriptor("mirrorPlaneCenter")
	node : PolyMirror = None
	pass
class MirrorPlaneCenterZPlug(Plug):
	parent : MirrorPlaneCenterPlug = PlugDescriptor("mirrorPlaneCenter")
	node : PolyMirror = None
	pass
class MirrorPlaneCenterPlug(Plug):
	mirrorPlaneCenterX_ : MirrorPlaneCenterXPlug = PlugDescriptor("mirrorPlaneCenterX")
	pcx_ : MirrorPlaneCenterXPlug = PlugDescriptor("mirrorPlaneCenterX")
	mirrorPlaneCenterY_ : MirrorPlaneCenterYPlug = PlugDescriptor("mirrorPlaneCenterY")
	pcy_ : MirrorPlaneCenterYPlug = PlugDescriptor("mirrorPlaneCenterY")
	mirrorPlaneCenterZ_ : MirrorPlaneCenterZPlug = PlugDescriptor("mirrorPlaneCenterZ")
	pcz_ : MirrorPlaneCenterZPlug = PlugDescriptor("mirrorPlaneCenterZ")
	node : PolyMirror = None
	pass
class MirrorPlaneRotateXPlug(Plug):
	parent : MirrorPlaneRotatePlug = PlugDescriptor("mirrorPlaneRotate")
	node : PolyMirror = None
	pass
class MirrorPlaneRotateYPlug(Plug):
	parent : MirrorPlaneRotatePlug = PlugDescriptor("mirrorPlaneRotate")
	node : PolyMirror = None
	pass
class MirrorPlaneRotateZPlug(Plug):
	parent : MirrorPlaneRotatePlug = PlugDescriptor("mirrorPlaneRotate")
	node : PolyMirror = None
	pass
class MirrorPlaneRotatePlug(Plug):
	mirrorPlaneRotateX_ : MirrorPlaneRotateXPlug = PlugDescriptor("mirrorPlaneRotateX")
	rx_ : MirrorPlaneRotateXPlug = PlugDescriptor("mirrorPlaneRotateX")
	mirrorPlaneRotateY_ : MirrorPlaneRotateYPlug = PlugDescriptor("mirrorPlaneRotateY")
	ry_ : MirrorPlaneRotateYPlug = PlugDescriptor("mirrorPlaneRotateY")
	mirrorPlaneRotateZ_ : MirrorPlaneRotateZPlug = PlugDescriptor("mirrorPlaneRotateZ")
	rz_ : MirrorPlaneRotateZPlug = PlugDescriptor("mirrorPlaneRotateZ")
	node : PolyMirror = None
	pass
class MirrorPositionPlug(Plug):
	node : PolyMirror = None
	pass
class PivotXPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : PolyMirror = None
	pass
class PivotYPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : PolyMirror = None
	pass
class PivotZPlug(Plug):
	parent : PivotPlug = PlugDescriptor("pivot")
	node : PolyMirror = None
	pass
class PivotPlug(Plug):
	pivotX_ : PivotXPlug = PlugDescriptor("pivotX")
	px_ : PivotXPlug = PlugDescriptor("pivotX")
	pivotY_ : PivotYPlug = PlugDescriptor("pivotY")
	py_ : PivotYPlug = PlugDescriptor("pivotY")
	pivotZ_ : PivotZPlug = PlugDescriptor("pivotZ")
	pz_ : PivotZPlug = PlugDescriptor("pivotZ")
	node : PolyMirror = None
	pass
class ScalePivotXPlug(Plug):
	parent : ScalePivotPlug = PlugDescriptor("scalePivot")
	node : PolyMirror = None
	pass
class ScalePivotYPlug(Plug):
	parent : ScalePivotPlug = PlugDescriptor("scalePivot")
	node : PolyMirror = None
	pass
class ScalePivotZPlug(Plug):
	parent : ScalePivotPlug = PlugDescriptor("scalePivot")
	node : PolyMirror = None
	pass
class ScalePivotPlug(Plug):
	scalePivotX_ : ScalePivotXPlug = PlugDescriptor("scalePivotX")
	spx_ : ScalePivotXPlug = PlugDescriptor("scalePivotX")
	scalePivotY_ : ScalePivotYPlug = PlugDescriptor("scalePivotY")
	spy_ : ScalePivotYPlug = PlugDescriptor("scalePivotY")
	scalePivotZ_ : ScalePivotZPlug = PlugDescriptor("scalePivotZ")
	spz_ : ScalePivotZPlug = PlugDescriptor("scalePivotZ")
	node : PolyMirror = None
	pass
class SmoothingAnglePlug(Plug):
	node : PolyMirror = None
	pass
class UserSpecifiedPivotPlug(Plug):
	node : PolyMirror = None
	pass
# endregion


# define node class
class PolyMirror(PolyModifierWorld):
	axis_ : AxisPlug = PlugDescriptor("axis")
	axisDirection_ : AxisDirectionPlug = PlugDescriptor("axisDirection")
	compId_ : CompIdPlug = PlugDescriptor("compId")
	cutMesh_ : CutMeshPlug = PlugDescriptor("cutMesh")
	direction_ : DirectionPlug = PlugDescriptor("direction")
	firstNewFace_ : FirstNewFacePlug = PlugDescriptor("firstNewFace")
	flipUVs_ : FlipUVsPlug = PlugDescriptor("flipUVs")
	keepVertexIDs_ : KeepVertexIDsPlug = PlugDescriptor("keepVertexIDs")
	lastNewFace_ : LastNewFacePlug = PlugDescriptor("lastNewFace")
	maya2017_ : Maya2017Plug = PlugDescriptor("maya2017")
	mergeMode_ : MergeModePlug = PlugDescriptor("mergeMode")
	mergeThreshold_ : MergeThresholdPlug = PlugDescriptor("mergeThreshold")
	mergeThresholdType_ : MergeThresholdTypePlug = PlugDescriptor("mergeThresholdType")
	mirrorAxis_ : MirrorAxisPlug = PlugDescriptor("mirrorAxis")
	mirrorPlaneCenterX_ : MirrorPlaneCenterXPlug = PlugDescriptor("mirrorPlaneCenterX")
	mirrorPlaneCenterY_ : MirrorPlaneCenterYPlug = PlugDescriptor("mirrorPlaneCenterY")
	mirrorPlaneCenterZ_ : MirrorPlaneCenterZPlug = PlugDescriptor("mirrorPlaneCenterZ")
	mirrorPlaneCenter_ : MirrorPlaneCenterPlug = PlugDescriptor("mirrorPlaneCenter")
	mirrorPlaneRotateX_ : MirrorPlaneRotateXPlug = PlugDescriptor("mirrorPlaneRotateX")
	mirrorPlaneRotateY_ : MirrorPlaneRotateYPlug = PlugDescriptor("mirrorPlaneRotateY")
	mirrorPlaneRotateZ_ : MirrorPlaneRotateZPlug = PlugDescriptor("mirrorPlaneRotateZ")
	mirrorPlaneRotate_ : MirrorPlaneRotatePlug = PlugDescriptor("mirrorPlaneRotate")
	mirrorPosition_ : MirrorPositionPlug = PlugDescriptor("mirrorPosition")
	pivotX_ : PivotXPlug = PlugDescriptor("pivotX")
	pivotY_ : PivotYPlug = PlugDescriptor("pivotY")
	pivotZ_ : PivotZPlug = PlugDescriptor("pivotZ")
	pivot_ : PivotPlug = PlugDescriptor("pivot")
	scalePivotX_ : ScalePivotXPlug = PlugDescriptor("scalePivotX")
	scalePivotY_ : ScalePivotYPlug = PlugDescriptor("scalePivotY")
	scalePivotZ_ : ScalePivotZPlug = PlugDescriptor("scalePivotZ")
	scalePivot_ : ScalePivotPlug = PlugDescriptor("scalePivot")
	smoothingAngle_ : SmoothingAnglePlug = PlugDescriptor("smoothingAngle")
	userSpecifiedPivot_ : UserSpecifiedPivotPlug = PlugDescriptor("userSpecifiedPivot")

	# node attributes

	typeName = "polyMirror"
	apiTypeInt = 958
	apiTypeStr = "kPolyMirror"
	typeIdInt = 1347242322
	MFnCls = om.MFnDependencyNode
	pass

