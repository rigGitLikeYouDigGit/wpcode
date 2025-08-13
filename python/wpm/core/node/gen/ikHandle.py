

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
class CheckSnappingFlagPlug(Plug):
	node : IkHandle = None
	pass
class DForwardAxisPlug(Plug):
	node : IkHandle = None
	pass
class DTwistControlEnablePlug(Plug):
	node : IkHandle = None
	pass
class DTwistRampBPlug(Plug):
	parent : DTwistRampPlug = PlugDescriptor("dTwistRamp")
	node : IkHandle = None
	pass
class DTwistRampGPlug(Plug):
	parent : DTwistRampPlug = PlugDescriptor("dTwistRamp")
	node : IkHandle = None
	pass
class DTwistRampRPlug(Plug):
	parent : DTwistRampPlug = PlugDescriptor("dTwistRamp")
	node : IkHandle = None
	pass
class DTwistRampPlug(Plug):
	dTwistRampB_ : DTwistRampBPlug = PlugDescriptor("dTwistRampB")
	dtrb_ : DTwistRampBPlug = PlugDescriptor("dTwistRampB")
	dTwistRampG_ : DTwistRampGPlug = PlugDescriptor("dTwistRampG")
	dtrg_ : DTwistRampGPlug = PlugDescriptor("dTwistRampG")
	dTwistRampR_ : DTwistRampRPlug = PlugDescriptor("dTwistRampR")
	dtrr_ : DTwistRampRPlug = PlugDescriptor("dTwistRampR")
	node : IkHandle = None
	pass
class DTwistRampMultPlug(Plug):
	node : IkHandle = None
	pass
class DTwistEndPlug(Plug):
	parent : DTwistStartEndPlug = PlugDescriptor("dTwistStartEnd")
	node : IkHandle = None
	pass
class DTwistStartPlug(Plug):
	parent : DTwistStartEndPlug = PlugDescriptor("dTwistStartEnd")
	node : IkHandle = None
	pass
class DTwistStartEndPlug(Plug):
	dTwistEnd_ : DTwistEndPlug = PlugDescriptor("dTwistEnd")
	dten_ : DTwistEndPlug = PlugDescriptor("dTwistEnd")
	dTwistStart_ : DTwistStartPlug = PlugDescriptor("dTwistStart")
	dtst_ : DTwistStartPlug = PlugDescriptor("dTwistStart")
	node : IkHandle = None
	pass
class DTwistValueTypePlug(Plug):
	node : IkHandle = None
	pass
class DWorldUpAxisPlug(Plug):
	node : IkHandle = None
	pass
class DWorldUpMatrixPlug(Plug):
	node : IkHandle = None
	pass
class DWorldUpMatrixEndPlug(Plug):
	node : IkHandle = None
	pass
class DWorldUpTypePlug(Plug):
	node : IkHandle = None
	pass
class DWorldUpVectorXPlug(Plug):
	parent : DWorldUpVectorPlug = PlugDescriptor("dWorldUpVector")
	node : IkHandle = None
	pass
class DWorldUpVectorYPlug(Plug):
	parent : DWorldUpVectorPlug = PlugDescriptor("dWorldUpVector")
	node : IkHandle = None
	pass
class DWorldUpVectorZPlug(Plug):
	parent : DWorldUpVectorPlug = PlugDescriptor("dWorldUpVector")
	node : IkHandle = None
	pass
class DWorldUpVectorPlug(Plug):
	dWorldUpVectorX_ : DWorldUpVectorXPlug = PlugDescriptor("dWorldUpVectorX")
	dwux_ : DWorldUpVectorXPlug = PlugDescriptor("dWorldUpVectorX")
	dWorldUpVectorY_ : DWorldUpVectorYPlug = PlugDescriptor("dWorldUpVectorY")
	dwuy_ : DWorldUpVectorYPlug = PlugDescriptor("dWorldUpVectorY")
	dWorldUpVectorZ_ : DWorldUpVectorZPlug = PlugDescriptor("dWorldUpVectorZ")
	dwuz_ : DWorldUpVectorZPlug = PlugDescriptor("dWorldUpVectorZ")
	node : IkHandle = None
	pass
class DWorldUpVectorEndXPlug(Plug):
	parent : DWorldUpVectorEndPlug = PlugDescriptor("dWorldUpVectorEnd")
	node : IkHandle = None
	pass
class DWorldUpVectorEndYPlug(Plug):
	parent : DWorldUpVectorEndPlug = PlugDescriptor("dWorldUpVectorEnd")
	node : IkHandle = None
	pass
class DWorldUpVectorEndZPlug(Plug):
	parent : DWorldUpVectorEndPlug = PlugDescriptor("dWorldUpVectorEnd")
	node : IkHandle = None
	pass
class DWorldUpVectorEndPlug(Plug):
	dWorldUpVectorEndX_ : DWorldUpVectorEndXPlug = PlugDescriptor("dWorldUpVectorEndX")
	dwvx_ : DWorldUpVectorEndXPlug = PlugDescriptor("dWorldUpVectorEndX")
	dWorldUpVectorEndY_ : DWorldUpVectorEndYPlug = PlugDescriptor("dWorldUpVectorEndY")
	dwvy_ : DWorldUpVectorEndYPlug = PlugDescriptor("dWorldUpVectorEndY")
	dWorldUpVectorEndZ_ : DWorldUpVectorEndZPlug = PlugDescriptor("dWorldUpVectorEndZ")
	dwvz_ : DWorldUpVectorEndZPlug = PlugDescriptor("dWorldUpVectorEndZ")
	node : IkHandle = None
	pass
class DofListPlug(Plug):
	node : IkHandle = None
	pass
class DofListDirtyFlagPlug(Plug):
	node : IkHandle = None
	pass
class EndEffectorPlug(Plug):
	node : IkHandle = None
	pass
class HandleDirtyFlagPlug(Plug):
	node : IkHandle = None
	pass
class IkBlendPlug(Plug):
	node : IkHandle = None
	pass
class IkFkManipulationPlug(Plug):
	node : IkHandle = None
	pass
class IkSolverPlug(Plug):
	node : IkHandle = None
	pass
class InCurvePlug(Plug):
	node : IkHandle = None
	pass
class OffsetPlug(Plug):
	node : IkHandle = None
	pass
class OwningHandleGroupPlug(Plug):
	node : IkHandle = None
	pass
class PoWeightPlug(Plug):
	node : IkHandle = None
	pass
class PoleVectorXPlug(Plug):
	parent : PoleVectorPlug = PlugDescriptor("poleVector")
	node : IkHandle = None
	pass
class PoleVectorYPlug(Plug):
	parent : PoleVectorPlug = PlugDescriptor("poleVector")
	node : IkHandle = None
	pass
class PoleVectorZPlug(Plug):
	parent : PoleVectorPlug = PlugDescriptor("poleVector")
	node : IkHandle = None
	pass
class PoleVectorPlug(Plug):
	poleVectorX_ : PoleVectorXPlug = PlugDescriptor("poleVectorX")
	pvx_ : PoleVectorXPlug = PlugDescriptor("poleVectorX")
	poleVectorY_ : PoleVectorYPlug = PlugDescriptor("poleVectorY")
	pvy_ : PoleVectorYPlug = PlugDescriptor("poleVectorY")
	poleVectorZ_ : PoleVectorZPlug = PlugDescriptor("poleVectorZ")
	pvz_ : PoleVectorZPlug = PlugDescriptor("poleVectorZ")
	node : IkHandle = None
	pass
class PriorityPlug(Plug):
	node : IkHandle = None
	pass
class RollPlug(Plug):
	node : IkHandle = None
	pass
class RootOnCurvePlug(Plug):
	node : IkHandle = None
	pass
class RootTwistModePlug(Plug):
	node : IkHandle = None
	pass
class SkeletonDirtyFlagPlug(Plug):
	node : IkHandle = None
	pass
class SnapEnablePlug(Plug):
	node : IkHandle = None
	pass
class SplineIkOldStylePlug(Plug):
	node : IkHandle = None
	pass
class StartJointPlug(Plug):
	node : IkHandle = None
	pass
class StickinessPlug(Plug):
	node : IkHandle = None
	pass
class TwistPlug(Plug):
	node : IkHandle = None
	pass
class TwistTypePlug(Plug):
	node : IkHandle = None
	pass
class WeightPlug(Plug):
	node : IkHandle = None
	pass
# endregion


# define node class
class IkHandle(Transform):
	checkSnappingFlag_ : CheckSnappingFlagPlug = PlugDescriptor("checkSnappingFlag")
	dForwardAxis_ : DForwardAxisPlug = PlugDescriptor("dForwardAxis")
	dTwistControlEnable_ : DTwistControlEnablePlug = PlugDescriptor("dTwistControlEnable")
	dTwistRampB_ : DTwistRampBPlug = PlugDescriptor("dTwistRampB")
	dTwistRampG_ : DTwistRampGPlug = PlugDescriptor("dTwistRampG")
	dTwistRampR_ : DTwistRampRPlug = PlugDescriptor("dTwistRampR")
	dTwistRamp_ : DTwistRampPlug = PlugDescriptor("dTwistRamp")
	dTwistRampMult_ : DTwistRampMultPlug = PlugDescriptor("dTwistRampMult")
	dTwistEnd_ : DTwistEndPlug = PlugDescriptor("dTwistEnd")
	dTwistStart_ : DTwistStartPlug = PlugDescriptor("dTwistStart")
	dTwistStartEnd_ : DTwistStartEndPlug = PlugDescriptor("dTwistStartEnd")
	dTwistValueType_ : DTwistValueTypePlug = PlugDescriptor("dTwistValueType")
	dWorldUpAxis_ : DWorldUpAxisPlug = PlugDescriptor("dWorldUpAxis")
	dWorldUpMatrix_ : DWorldUpMatrixPlug = PlugDescriptor("dWorldUpMatrix")
	dWorldUpMatrixEnd_ : DWorldUpMatrixEndPlug = PlugDescriptor("dWorldUpMatrixEnd")
	dWorldUpType_ : DWorldUpTypePlug = PlugDescriptor("dWorldUpType")
	dWorldUpVectorX_ : DWorldUpVectorXPlug = PlugDescriptor("dWorldUpVectorX")
	dWorldUpVectorY_ : DWorldUpVectorYPlug = PlugDescriptor("dWorldUpVectorY")
	dWorldUpVectorZ_ : DWorldUpVectorZPlug = PlugDescriptor("dWorldUpVectorZ")
	dWorldUpVector_ : DWorldUpVectorPlug = PlugDescriptor("dWorldUpVector")
	dWorldUpVectorEndX_ : DWorldUpVectorEndXPlug = PlugDescriptor("dWorldUpVectorEndX")
	dWorldUpVectorEndY_ : DWorldUpVectorEndYPlug = PlugDescriptor("dWorldUpVectorEndY")
	dWorldUpVectorEndZ_ : DWorldUpVectorEndZPlug = PlugDescriptor("dWorldUpVectorEndZ")
	dWorldUpVectorEnd_ : DWorldUpVectorEndPlug = PlugDescriptor("dWorldUpVectorEnd")
	dofList_ : DofListPlug = PlugDescriptor("dofList")
	dofListDirtyFlag_ : DofListDirtyFlagPlug = PlugDescriptor("dofListDirtyFlag")
	endEffector_ : EndEffectorPlug = PlugDescriptor("endEffector")
	handleDirtyFlag_ : HandleDirtyFlagPlug = PlugDescriptor("handleDirtyFlag")
	ikBlend_ : IkBlendPlug = PlugDescriptor("ikBlend")
	ikFkManipulation_ : IkFkManipulationPlug = PlugDescriptor("ikFkManipulation")
	ikSolver_ : IkSolverPlug = PlugDescriptor("ikSolver")
	inCurve_ : InCurvePlug = PlugDescriptor("inCurve")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	owningHandleGroup_ : OwningHandleGroupPlug = PlugDescriptor("owningHandleGroup")
	poWeight_ : PoWeightPlug = PlugDescriptor("poWeight")
	poleVectorX_ : PoleVectorXPlug = PlugDescriptor("poleVectorX")
	poleVectorY_ : PoleVectorYPlug = PlugDescriptor("poleVectorY")
	poleVectorZ_ : PoleVectorZPlug = PlugDescriptor("poleVectorZ")
	poleVector_ : PoleVectorPlug = PlugDescriptor("poleVector")
	priority_ : PriorityPlug = PlugDescriptor("priority")
	roll_ : RollPlug = PlugDescriptor("roll")
	rootOnCurve_ : RootOnCurvePlug = PlugDescriptor("rootOnCurve")
	rootTwistMode_ : RootTwistModePlug = PlugDescriptor("rootTwistMode")
	skeletonDirtyFlag_ : SkeletonDirtyFlagPlug = PlugDescriptor("skeletonDirtyFlag")
	snapEnable_ : SnapEnablePlug = PlugDescriptor("snapEnable")
	splineIkOldStyle_ : SplineIkOldStylePlug = PlugDescriptor("splineIkOldStyle")
	startJoint_ : StartJointPlug = PlugDescriptor("startJoint")
	stickiness_ : StickinessPlug = PlugDescriptor("stickiness")
	twist_ : TwistPlug = PlugDescriptor("twist")
	twistType_ : TwistTypePlug = PlugDescriptor("twistType")
	weight_ : WeightPlug = PlugDescriptor("weight")

	# node attributes

	typeName = "ikHandle"
	apiTypeInt = 120
	apiTypeStr = "kIkHandle"
	typeIdInt = 1263027276
	MFnCls = om.MFnTransform
	pass

