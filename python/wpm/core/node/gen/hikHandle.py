

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
IkHandle = retriever.getNodeCls("IkHandle")
assert IkHandle
if T.TYPE_CHECKING:
	from .. import IkHandle

# add node doc



# region plug type defs
class ActivatePlug(Plug):
	node : HikHandle = None
	pass
class ChestPullPlug(Plug):
	parent : ChestPlug = PlugDescriptor("chest")
	node : HikHandle = None
	pass
class ChestPlug(Plug):
	chestPull_ : ChestPullPlug = PlugDescriptor("chestPull")
	rcp_ : ChestPullPlug = PlugDescriptor("chestPull")
	node : HikHandle = None
	pass
class HandBackPlug(Plug):
	parent : ContactsPositionPlug = PlugDescriptor("contactsPosition")
	node : HikHandle = None
	pass
class HandFrontPlug(Plug):
	parent : ContactsPositionPlug = PlugDescriptor("contactsPosition")
	node : HikHandle = None
	pass
class HandHeightPlug(Plug):
	parent : ContactsPositionPlug = PlugDescriptor("contactsPosition")
	node : HikHandle = None
	pass
class HandInSidePlug(Plug):
	parent : ContactsPositionPlug = PlugDescriptor("contactsPosition")
	node : HikHandle = None
	pass
class HandMiddlePlug(Plug):
	parent : ContactsPositionPlug = PlugDescriptor("contactsPosition")
	node : HikHandle = None
	pass
class HandOutSidePlug(Plug):
	parent : ContactsPositionPlug = PlugDescriptor("contactsPosition")
	node : HikHandle = None
	pass
class ContactsPositionPlug(Plug):
	handBack_ : HandBackPlug = PlugDescriptor("handBack")
	hb_ : HandBackPlug = PlugDescriptor("handBack")
	handFront_ : HandFrontPlug = PlugDescriptor("handFront")
	hf_ : HandFrontPlug = PlugDescriptor("handFront")
	handHeight_ : HandHeightPlug = PlugDescriptor("handHeight")
	hh_ : HandHeightPlug = PlugDescriptor("handHeight")
	handInSide_ : HandInSidePlug = PlugDescriptor("handInSide")
	his_ : HandInSidePlug = PlugDescriptor("handInSide")
	handMiddle_ : HandMiddlePlug = PlugDescriptor("handMiddle")
	hm_ : HandMiddlePlug = PlugDescriptor("handMiddle")
	handOutSide_ : HandOutSidePlug = PlugDescriptor("handOutSide")
	hos_ : HandOutSidePlug = PlugDescriptor("handOutSide")
	node : HikHandle = None
	pass
class ConvertScalePlug(Plug):
	node : HikHandle = None
	pass
class DefaultMatrixPlug(Plug):
	node : HikHandle = None
	pass
class EffectorsPlug(Plug):
	node : HikHandle = None
	pass
class PullIterationCountPlug(Plug):
	parent : ExtraPlug = PlugDescriptor("extra")
	node : HikHandle = None
	pass
class ExtraPlug(Plug):
	pullIterationCount_ : PullIterationCountPlug = PlugDescriptor("pullIterationCount")
	pic_ : PullIterationCountPlug = PlugDescriptor("pullIterationCount")
	node : HikHandle = None
	pass
class FootBackPlug(Plug):
	parent : FeetContactPositionPlug = PlugDescriptor("feetContactPosition")
	node : HikHandle = None
	pass
class FootFrontPlug(Plug):
	parent : FeetContactPositionPlug = PlugDescriptor("feetContactPosition")
	node : HikHandle = None
	pass
class FootHeightPlug(Plug):
	parent : FeetContactPositionPlug = PlugDescriptor("feetContactPosition")
	node : HikHandle = None
	pass
class FootInSidePlug(Plug):
	parent : FeetContactPositionPlug = PlugDescriptor("feetContactPosition")
	node : HikHandle = None
	pass
class FootMiddlePlug(Plug):
	parent : FeetContactPositionPlug = PlugDescriptor("feetContactPosition")
	node : HikHandle = None
	pass
class FootOutSidePlug(Plug):
	parent : FeetContactPositionPlug = PlugDescriptor("feetContactPosition")
	node : HikHandle = None
	pass
class FeetContactPositionPlug(Plug):
	footBack_ : FootBackPlug = PlugDescriptor("footBack")
	fra_ : FootBackPlug = PlugDescriptor("footBack")
	footFront_ : FootFrontPlug = PlugDescriptor("footFront")
	ffm_ : FootFrontPlug = PlugDescriptor("footFront")
	footHeight_ : FootHeightPlug = PlugDescriptor("footHeight")
	fh_ : FootHeightPlug = PlugDescriptor("footHeight")
	footInSide_ : FootInSidePlug = PlugDescriptor("footInSide")
	fia_ : FootInSidePlug = PlugDescriptor("footInSide")
	footMiddle_ : FootMiddlePlug = PlugDescriptor("footMiddle")
	fma_ : FootMiddlePlug = PlugDescriptor("footMiddle")
	footOutSide_ : FootOutSidePlug = PlugDescriptor("footOutSide")
	foa_ : FootOutSidePlug = PlugDescriptor("footOutSide")
	node : HikHandle = None
	pass
class FeetContactStiffnessPlug(Plug):
	parent : FeetFloorContactSetupPlug = PlugDescriptor("feetFloorContactSetup")
	node : HikHandle = None
	pass
class FeetContactTypePlug(Plug):
	parent : FeetFloorContactSetupPlug = PlugDescriptor("feetFloorContactSetup")
	node : HikHandle = None
	pass
class FeetFloorPivotPlug(Plug):
	parent : FeetFloorContactSetupPlug = PlugDescriptor("feetFloorContactSetup")
	node : HikHandle = None
	pass
class FeetFloorContactSetupPlug(Plug):
	feetContactStiffness_ : FeetContactStiffnessPlug = PlugDescriptor("feetContactStiffness")
	fcs_ : FeetContactStiffnessPlug = PlugDescriptor("feetContactStiffness")
	feetContactType_ : FeetContactTypePlug = PlugDescriptor("feetContactType")
	fct_ : FeetContactTypePlug = PlugDescriptor("feetContactType")
	feetFloorPivot_ : FeetFloorPivotPlug = PlugDescriptor("feetFloorPivot")
	fpv_ : FeetFloorPivotPlug = PlugDescriptor("feetFloorPivot")
	node : HikHandle = None
	pass
class LeftHandExtraFingerTipPlug(Plug):
	parent : FingerTipsSizesPlug = PlugDescriptor("fingerTipsSizes")
	node : HikHandle = None
	pass
class LeftHandIndexTipPlug(Plug):
	parent : FingerTipsSizesPlug = PlugDescriptor("fingerTipsSizes")
	node : HikHandle = None
	pass
class LeftHandMiddleTipPlug(Plug):
	parent : FingerTipsSizesPlug = PlugDescriptor("fingerTipsSizes")
	node : HikHandle = None
	pass
class LeftHandPinkyTipPlug(Plug):
	parent : FingerTipsSizesPlug = PlugDescriptor("fingerTipsSizes")
	node : HikHandle = None
	pass
class LeftHandRingTipPlug(Plug):
	parent : FingerTipsSizesPlug = PlugDescriptor("fingerTipsSizes")
	node : HikHandle = None
	pass
class LeftHandThumbTipPlug(Plug):
	parent : FingerTipsSizesPlug = PlugDescriptor("fingerTipsSizes")
	node : HikHandle = None
	pass
class RightHandExtraFingerTipPlug(Plug):
	parent : FingerTipsSizesPlug = PlugDescriptor("fingerTipsSizes")
	node : HikHandle = None
	pass
class RightHandIndexTipPlug(Plug):
	parent : FingerTipsSizesPlug = PlugDescriptor("fingerTipsSizes")
	node : HikHandle = None
	pass
class RightHandMiddleTipPlug(Plug):
	parent : FingerTipsSizesPlug = PlugDescriptor("fingerTipsSizes")
	node : HikHandle = None
	pass
class RightHandPinkyTipPlug(Plug):
	parent : FingerTipsSizesPlug = PlugDescriptor("fingerTipsSizes")
	node : HikHandle = None
	pass
class RightHandRingTipPlug(Plug):
	parent : FingerTipsSizesPlug = PlugDescriptor("fingerTipsSizes")
	node : HikHandle = None
	pass
class RightHandThumbTipPlug(Plug):
	parent : FingerTipsSizesPlug = PlugDescriptor("fingerTipsSizes")
	node : HikHandle = None
	pass
class FingerTipsSizesPlug(Plug):
	leftHandExtraFingerTip_ : LeftHandExtraFingerTipPlug = PlugDescriptor("leftHandExtraFingerTip")
	lxt_ : LeftHandExtraFingerTipPlug = PlugDescriptor("leftHandExtraFingerTip")
	leftHandIndexTip_ : LeftHandIndexTipPlug = PlugDescriptor("leftHandIndexTip")
	lit_ : LeftHandIndexTipPlug = PlugDescriptor("leftHandIndexTip")
	leftHandMiddleTip_ : LeftHandMiddleTipPlug = PlugDescriptor("leftHandMiddleTip")
	lmt_ : LeftHandMiddleTipPlug = PlugDescriptor("leftHandMiddleTip")
	leftHandPinkyTip_ : LeftHandPinkyTipPlug = PlugDescriptor("leftHandPinkyTip")
	lpt_ : LeftHandPinkyTipPlug = PlugDescriptor("leftHandPinkyTip")
	leftHandRingTip_ : LeftHandRingTipPlug = PlugDescriptor("leftHandRingTip")
	lrt_ : LeftHandRingTipPlug = PlugDescriptor("leftHandRingTip")
	leftHandThumbTip_ : LeftHandThumbTipPlug = PlugDescriptor("leftHandThumbTip")
	ltt_ : LeftHandThumbTipPlug = PlugDescriptor("leftHandThumbTip")
	rightHandExtraFingerTip_ : RightHandExtraFingerTipPlug = PlugDescriptor("rightHandExtraFingerTip")
	rxt_ : RightHandExtraFingerTipPlug = PlugDescriptor("rightHandExtraFingerTip")
	rightHandIndexTip_ : RightHandIndexTipPlug = PlugDescriptor("rightHandIndexTip")
	rit_ : RightHandIndexTipPlug = PlugDescriptor("rightHandIndexTip")
	rightHandMiddleTip_ : RightHandMiddleTipPlug = PlugDescriptor("rightHandMiddleTip")
	rmt_ : RightHandMiddleTipPlug = PlugDescriptor("rightHandMiddleTip")
	rightHandPinkyTip_ : RightHandPinkyTipPlug = PlugDescriptor("rightHandPinkyTip")
	rpp_ : RightHandPinkyTipPlug = PlugDescriptor("rightHandPinkyTip")
	rightHandRingTip_ : RightHandRingTipPlug = PlugDescriptor("rightHandRingTip")
	rrt_ : RightHandRingTipPlug = PlugDescriptor("rightHandRingTip")
	rightHandThumbTip_ : RightHandThumbTipPlug = PlugDescriptor("rightHandThumbTip")
	rtt_ : RightHandThumbTipPlug = PlugDescriptor("rightHandThumbTip")
	node : HikHandle = None
	pass
class FingersContactRollStiffnessPlug(Plug):
	parent : FingersFloorContactSetupPlug = PlugDescriptor("fingersFloorContactSetup")
	node : HikHandle = None
	pass
class FingersContactTypePlug(Plug):
	parent : FingersFloorContactSetupPlug = PlugDescriptor("fingersFloorContactSetup")
	node : HikHandle = None
	pass
class FingersFloorContactSetupPlug(Plug):
	fingersContactRollStiffness_ : FingersContactRollStiffnessPlug = PlugDescriptor("fingersContactRollStiffness")
	hcr_ : FingersContactRollStiffnessPlug = PlugDescriptor("fingersContactRollStiffness")
	fingersContactType_ : FingersContactTypePlug = PlugDescriptor("fingersContactType")
	fcm_ : FingersContactTypePlug = PlugDescriptor("fingersContactType")
	node : HikHandle = None
	pass
class FkjointsPlug(Plug):
	node : HikHandle = None
	pass
class FkmatrixPlug(Plug):
	node : HikHandle = None
	pass
class FeetFloorContactPlug(Plug):
	parent : FloorContactsPlug = PlugDescriptor("floorContacts")
	node : HikHandle = None
	pass
class FingersFloorContactPlug(Plug):
	parent : FloorContactsPlug = PlugDescriptor("floorContacts")
	node : HikHandle = None
	pass
class HandsFloorContactPlug(Plug):
	parent : FloorContactsPlug = PlugDescriptor("floorContacts")
	node : HikHandle = None
	pass
class ToesFloorContactPlug(Plug):
	parent : FloorContactsPlug = PlugDescriptor("floorContacts")
	node : HikHandle = None
	pass
class FloorContactsPlug(Plug):
	feetFloorContact_ : FeetFloorContactPlug = PlugDescriptor("feetFloorContact")
	fec_ : FeetFloorContactPlug = PlugDescriptor("feetFloorContact")
	fingersFloorContact_ : FingersFloorContactPlug = PlugDescriptor("fingersFloorContact")
	fic_ : FingersFloorContactPlug = PlugDescriptor("fingersFloorContact")
	handsFloorContact_ : HandsFloorContactPlug = PlugDescriptor("handsFloorContact")
	hfc_ : HandsFloorContactPlug = PlugDescriptor("handsFloorContact")
	toesFloorContact_ : ToesFloorContactPlug = PlugDescriptor("toesFloorContact")
	tfc_ : ToesFloorContactPlug = PlugDescriptor("toesFloorContact")
	node : HikHandle = None
	pass
class HandsContactStiffnessPlug(Plug):
	parent : HandsFloorContactSetupPlug = PlugDescriptor("handsFloorContactSetup")
	node : HikHandle = None
	pass
class HandsContactTypePlug(Plug):
	parent : HandsFloorContactSetupPlug = PlugDescriptor("handsFloorContactSetup")
	node : HikHandle = None
	pass
class HandsFloorPivotPlug(Plug):
	parent : HandsFloorContactSetupPlug = PlugDescriptor("handsFloorContactSetup")
	node : HikHandle = None
	pass
class HandsFloorContactSetupPlug(Plug):
	handsContactStiffness_ : HandsContactStiffnessPlug = PlugDescriptor("handsContactStiffness")
	hcs_ : HandsContactStiffnessPlug = PlugDescriptor("handsContactStiffness")
	handsContactType_ : HandsContactTypePlug = PlugDescriptor("handsContactType")
	hct_ : HandsContactTypePlug = PlugDescriptor("handsContactType")
	handsFloorPivot_ : HandsFloorPivotPlug = PlugDescriptor("handsFloorPivot")
	hfp_ : HandsFloorPivotPlug = PlugDescriptor("handsFloorPivot")
	node : HikHandle = None
	pass
class HeadPullPlug(Plug):
	parent : HeadPlug = PlugDescriptor("head")
	node : HikHandle = None
	pass
class HeadPlug(Plug):
	headPull_ : HeadPullPlug = PlugDescriptor("headPull")
	phd_ : HeadPullPlug = PlugDescriptor("headPull")
	node : HikHandle = None
	pass
class HipsPullPlug(Plug):
	parent : HipsPlug = PlugDescriptor("hips")
	node : HikHandle = None
	pass
class HipsPlug(Plug):
	hipsPull_ : HipsPullPlug = PlugDescriptor("hipsPull")
	chp_ : HipsPullPlug = PlugDescriptor("hipsPull")
	node : HikHandle = None
	pass
class JointsPlug(Plug):
	node : HikHandle = None
	pass
class LeftElbowKillPitchPlug(Plug):
	parent : KillPitchPlug = PlugDescriptor("killPitch")
	node : HikHandle = None
	pass
class LeftKneeKillPitchPlug(Plug):
	parent : KillPitchPlug = PlugDescriptor("killPitch")
	node : HikHandle = None
	pass
class RightElbowKillPitchPlug(Plug):
	parent : KillPitchPlug = PlugDescriptor("killPitch")
	node : HikHandle = None
	pass
class RightKneeKillPitchPlug(Plug):
	parent : KillPitchPlug = PlugDescriptor("killPitch")
	node : HikHandle = None
	pass
class KillPitchPlug(Plug):
	leftElbowKillPitch_ : LeftElbowKillPitchPlug = PlugDescriptor("leftElbowKillPitch")
	lek_ : LeftElbowKillPitchPlug = PlugDescriptor("leftElbowKillPitch")
	leftKneeKillPitch_ : LeftKneeKillPitchPlug = PlugDescriptor("leftKneeKillPitch")
	lkk_ : LeftKneeKillPitchPlug = PlugDescriptor("leftKneeKillPitch")
	rightElbowKillPitch_ : RightElbowKillPitchPlug = PlugDescriptor("rightElbowKillPitch")
	rek_ : RightElbowKillPitchPlug = PlugDescriptor("rightElbowKillPitch")
	rightKneeKillPitch_ : RightKneeKillPitchPlug = PlugDescriptor("rightKneeKillPitch")
	rkk_ : RightKneeKillPitchPlug = PlugDescriptor("rightKneeKillPitch")
	node : HikHandle = None
	pass
class LeftElbowPullPlug(Plug):
	parent : LeftArmPlug = PlugDescriptor("leftArm")
	node : HikHandle = None
	pass
class LeftFingerBasePullPlug(Plug):
	parent : LeftArmPlug = PlugDescriptor("leftArm")
	node : HikHandle = None
	pass
class LeftHandPullChestPlug(Plug):
	parent : LeftArmPlug = PlugDescriptor("leftArm")
	node : HikHandle = None
	pass
class LeftHandPullHipsPlug(Plug):
	parent : LeftArmPlug = PlugDescriptor("leftArm")
	node : HikHandle = None
	pass
class LeftArmPlug(Plug):
	leftElbowPull_ : LeftElbowPullPlug = PlugDescriptor("leftElbowPull")
	ple_ : LeftElbowPullPlug = PlugDescriptor("leftElbowPull")
	leftFingerBasePull_ : LeftFingerBasePullPlug = PlugDescriptor("leftFingerBasePull")
	plb_ : LeftFingerBasePullPlug = PlugDescriptor("leftFingerBasePull")
	leftHandPullChest_ : LeftHandPullChestPlug = PlugDescriptor("leftHandPullChest")
	cpl_ : LeftHandPullChestPlug = PlugDescriptor("leftHandPullChest")
	leftHandPullHips_ : LeftHandPullHipsPlug = PlugDescriptor("leftHandPullHips")
	plh_ : LeftHandPullHipsPlug = PlugDescriptor("leftHandPullHips")
	node : HikHandle = None
	pass
class LeftFootGroundPlanePlug(Plug):
	node : HikHandle = None
	pass
class LeftFootOrientedGroundPlanePlug(Plug):
	node : HikHandle = None
	pass
class LeftHandGroundPlanePlug(Plug):
	node : HikHandle = None
	pass
class LeftHandOrientedGroundPlanePlug(Plug):
	node : HikHandle = None
	pass
class LeftFootPullPlug(Plug):
	parent : LeftLegPlug = PlugDescriptor("leftLeg")
	node : HikHandle = None
	pass
class LeftKneePullPlug(Plug):
	parent : LeftLegPlug = PlugDescriptor("leftLeg")
	node : HikHandle = None
	pass
class LeftToeBasePullPlug(Plug):
	parent : LeftLegPlug = PlugDescriptor("leftLeg")
	node : HikHandle = None
	pass
class LeftLegPlug(Plug):
	leftFootPull_ : LeftFootPullPlug = PlugDescriptor("leftFootPull")
	plf_ : LeftFootPullPlug = PlugDescriptor("leftFootPull")
	leftKneePull_ : LeftKneePullPlug = PlugDescriptor("leftKneePull")
	plk_ : LeftKneePullPlug = PlugDescriptor("leftKneePull")
	leftToeBasePull_ : LeftToeBasePullPlug = PlugDescriptor("leftToeBasePull")
	plt_ : LeftToeBasePullPlug = PlugDescriptor("leftToeBasePull")
	node : HikHandle = None
	pass
class PropertyChangedPlug(Plug):
	node : HikHandle = None
	pass
class RightElbowPullPlug(Plug):
	parent : RightArmPlug = PlugDescriptor("rightArm")
	node : HikHandle = None
	pass
class RightFingerBasePullPlug(Plug):
	parent : RightArmPlug = PlugDescriptor("rightArm")
	node : HikHandle = None
	pass
class RightHandPullChestPlug(Plug):
	parent : RightArmPlug = PlugDescriptor("rightArm")
	node : HikHandle = None
	pass
class RightHandPullHipsPlug(Plug):
	parent : RightArmPlug = PlugDescriptor("rightArm")
	node : HikHandle = None
	pass
class RightArmPlug(Plug):
	rightElbowPull_ : RightElbowPullPlug = PlugDescriptor("rightElbowPull")
	pre_ : RightElbowPullPlug = PlugDescriptor("rightElbowPull")
	rightFingerBasePull_ : RightFingerBasePullPlug = PlugDescriptor("rightFingerBasePull")
	prb_ : RightFingerBasePullPlug = PlugDescriptor("rightFingerBasePull")
	rightHandPullChest_ : RightHandPullChestPlug = PlugDescriptor("rightHandPullChest")
	cpr_ : RightHandPullChestPlug = PlugDescriptor("rightHandPullChest")
	rightHandPullHips_ : RightHandPullHipsPlug = PlugDescriptor("rightHandPullHips")
	prh_ : RightHandPullHipsPlug = PlugDescriptor("rightHandPullHips")
	node : HikHandle = None
	pass
class RightFootGroundPlanePlug(Plug):
	node : HikHandle = None
	pass
class RightFootOrientedGroundPlanePlug(Plug):
	node : HikHandle = None
	pass
class RightHandGroundPlanePlug(Plug):
	node : HikHandle = None
	pass
class RightHandOrientedGroundPlanePlug(Plug):
	node : HikHandle = None
	pass
class RightFootPullPlug(Plug):
	parent : RightLegPlug = PlugDescriptor("rightLeg")
	node : HikHandle = None
	pass
class RightKneePullPlug(Plug):
	parent : RightLegPlug = PlugDescriptor("rightLeg")
	node : HikHandle = None
	pass
class RightToeBasePullPlug(Plug):
	parent : RightLegPlug = PlugDescriptor("rightLeg")
	node : HikHandle = None
	pass
class RightLegPlug(Plug):
	rightFootPull_ : RightFootPullPlug = PlugDescriptor("rightFootPull")
	prf_ : RightFootPullPlug = PlugDescriptor("rightFootPull")
	rightKneePull_ : RightKneePullPlug = PlugDescriptor("rightKneePull")
	prk_ : RightKneePullPlug = PlugDescriptor("rightKneePull")
	rightToeBasePull_ : RightToeBasePullPlug = PlugDescriptor("rightToeBasePull")
	prt_ : RightToeBasePullPlug = PlugDescriptor("rightToeBasePull")
	node : HikHandle = None
	pass
class LeftArmRollPlug(Plug):
	parent : RollExtractionPlug = PlugDescriptor("rollExtraction")
	node : HikHandle = None
	pass
class LeftArmRollModePlug(Plug):
	parent : RollExtractionPlug = PlugDescriptor("rollExtraction")
	node : HikHandle = None
	pass
class LeftForeArmRollPlug(Plug):
	parent : RollExtractionPlug = PlugDescriptor("rollExtraction")
	node : HikHandle = None
	pass
class LeftForeArmRollModePlug(Plug):
	parent : RollExtractionPlug = PlugDescriptor("rollExtraction")
	node : HikHandle = None
	pass
class LeftLegRollPlug(Plug):
	parent : RollExtractionPlug = PlugDescriptor("rollExtraction")
	node : HikHandle = None
	pass
class LeftLegRollModePlug(Plug):
	parent : RollExtractionPlug = PlugDescriptor("rollExtraction")
	node : HikHandle = None
	pass
class LeftUpLegRollPlug(Plug):
	parent : RollExtractionPlug = PlugDescriptor("rollExtraction")
	node : HikHandle = None
	pass
class LeftUpLegRollModePlug(Plug):
	parent : RollExtractionPlug = PlugDescriptor("rollExtraction")
	node : HikHandle = None
	pass
class RightArmRollPlug(Plug):
	parent : RollExtractionPlug = PlugDescriptor("rollExtraction")
	node : HikHandle = None
	pass
class RightArmRollModePlug(Plug):
	parent : RollExtractionPlug = PlugDescriptor("rollExtraction")
	node : HikHandle = None
	pass
class RightForeArmRollPlug(Plug):
	parent : RollExtractionPlug = PlugDescriptor("rollExtraction")
	node : HikHandle = None
	pass
class RightForeArmRollModePlug(Plug):
	parent : RollExtractionPlug = PlugDescriptor("rollExtraction")
	node : HikHandle = None
	pass
class RightLegRollPlug(Plug):
	parent : RollExtractionPlug = PlugDescriptor("rollExtraction")
	node : HikHandle = None
	pass
class RightLegRollModePlug(Plug):
	parent : RollExtractionPlug = PlugDescriptor("rollExtraction")
	node : HikHandle = None
	pass
class RightUpLegRollPlug(Plug):
	parent : RollExtractionPlug = PlugDescriptor("rollExtraction")
	node : HikHandle = None
	pass
class RightUpLegRollModePlug(Plug):
	parent : RollExtractionPlug = PlugDescriptor("rollExtraction")
	node : HikHandle = None
	pass
class RollExtractionModePlug(Plug):
	parent : RollExtractionPlug = PlugDescriptor("rollExtraction")
	node : HikHandle = None
	pass
class RollExtractionPlug(Plug):
	leftArmRoll_ : LeftArmRollPlug = PlugDescriptor("leftArmRoll")
	lar_ : LeftArmRollPlug = PlugDescriptor("leftArmRoll")
	leftArmRollMode_ : LeftArmRollModePlug = PlugDescriptor("leftArmRollMode")
	larm_ : LeftArmRollModePlug = PlugDescriptor("leftArmRollMode")
	leftForeArmRoll_ : LeftForeArmRollPlug = PlugDescriptor("leftForeArmRoll")
	lfr_ : LeftForeArmRollPlug = PlugDescriptor("leftForeArmRoll")
	leftForeArmRollMode_ : LeftForeArmRollModePlug = PlugDescriptor("leftForeArmRollMode")
	lfrm_ : LeftForeArmRollModePlug = PlugDescriptor("leftForeArmRollMode")
	leftLegRoll_ : LeftLegRollPlug = PlugDescriptor("leftLegRoll")
	llr_ : LeftLegRollPlug = PlugDescriptor("leftLegRoll")
	leftLegRollMode_ : LeftLegRollModePlug = PlugDescriptor("leftLegRollMode")
	llrm_ : LeftLegRollModePlug = PlugDescriptor("leftLegRollMode")
	leftUpLegRoll_ : LeftUpLegRollPlug = PlugDescriptor("leftUpLegRoll")
	lur_ : LeftUpLegRollPlug = PlugDescriptor("leftUpLegRoll")
	leftUpLegRollMode_ : LeftUpLegRollModePlug = PlugDescriptor("leftUpLegRollMode")
	lurm_ : LeftUpLegRollModePlug = PlugDescriptor("leftUpLegRollMode")
	rightArmRoll_ : RightArmRollPlug = PlugDescriptor("rightArmRoll")
	rar_ : RightArmRollPlug = PlugDescriptor("rightArmRoll")
	rightArmRollMode_ : RightArmRollModePlug = PlugDescriptor("rightArmRollMode")
	rarm_ : RightArmRollModePlug = PlugDescriptor("rightArmRollMode")
	rightForeArmRoll_ : RightForeArmRollPlug = PlugDescriptor("rightForeArmRoll")
	rfr_ : RightForeArmRollPlug = PlugDescriptor("rightForeArmRoll")
	rightForeArmRollMode_ : RightForeArmRollModePlug = PlugDescriptor("rightForeArmRollMode")
	rfrm_ : RightForeArmRollModePlug = PlugDescriptor("rightForeArmRollMode")
	rightLegRoll_ : RightLegRollPlug = PlugDescriptor("rightLegRoll")
	rlro_ : RightLegRollPlug = PlugDescriptor("rightLegRoll")
	rightLegRollMode_ : RightLegRollModePlug = PlugDescriptor("rightLegRollMode")
	rlrm_ : RightLegRollModePlug = PlugDescriptor("rightLegRollMode")
	rightUpLegRoll_ : RightUpLegRollPlug = PlugDescriptor("rightUpLegRoll")
	rur_ : RightUpLegRollPlug = PlugDescriptor("rightUpLegRoll")
	rightUpLegRollMode_ : RightUpLegRollModePlug = PlugDescriptor("rightUpLegRollMode")
	rurm_ : RightUpLegRollModePlug = PlugDescriptor("rightUpLegRollMode")
	rollExtractionMode_ : RollExtractionModePlug = PlugDescriptor("rollExtractionMode")
	rem_ : RollExtractionModePlug = PlugDescriptor("rollExtractionMode")
	node : HikHandle = None
	pass
class ExpertModePlug(Plug):
	parent : SolvingPlug = PlugDescriptor("solving")
	node : HikHandle = None
	pass
class HipTranslationModePlug(Plug):
	parent : SolvingPlug = PlugDescriptor("solving")
	node : HikHandle = None
	pass
class PostureTypePlug(Plug):
	parent : SolvingPlug = PlugDescriptor("solving")
	node : HikHandle = None
	pass
class RealisticShoulderSolvingPlug(Plug):
	parent : SolvingPlug = PlugDescriptor("solving")
	node : HikHandle = None
	pass
class SolveFingersPlug(Plug):
	parent : SolvingPlug = PlugDescriptor("solving")
	node : HikHandle = None
	pass
class SolvingPlug(Plug):
	expertMode_ : ExpertModePlug = PlugDescriptor("expertMode")
	exp_ : ExpertModePlug = PlugDescriptor("expertMode")
	hipTranslationMode_ : HipTranslationModePlug = PlugDescriptor("hipTranslationMode")
	htm_ : HipTranslationModePlug = PlugDescriptor("hipTranslationMode")
	postureType_ : PostureTypePlug = PlugDescriptor("postureType")
	pt_ : PostureTypePlug = PlugDescriptor("postureType")
	realisticShoulderSolving_ : RealisticShoulderSolvingPlug = PlugDescriptor("realisticShoulderSolving")
	rss_ : RealisticShoulderSolvingPlug = PlugDescriptor("realisticShoulderSolving")
	solveFingers_ : SolveFingersPlug = PlugDescriptor("solveFingers")
	sf_ : SolveFingersPlug = PlugDescriptor("solveFingers")
	node : HikHandle = None
	pass
class StancePoseMatrixPlug(Plug):
	node : HikHandle = None
	pass
class ChestStiffnessPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : HikHandle = None
	pass
class HipsEnforceGravityPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : HikHandle = None
	pass
class HipsStiffnessPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : HikHandle = None
	pass
class LeftArmStiffnessPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : HikHandle = None
	pass
class LeftElbowCompressionFactorPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : HikHandle = None
	pass
class LeftElbowMaxExtensionPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : HikHandle = None
	pass
class LeftKneeCompressionFactorPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : HikHandle = None
	pass
class LeftKneeMaxExtensionPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : HikHandle = None
	pass
class LeftLegStiffnessPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : HikHandle = None
	pass
class LeftShoulderStiffnessPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : HikHandle = None
	pass
class NeckStiffnessPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : HikHandle = None
	pass
class RightArmStiffnessPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : HikHandle = None
	pass
class RightElbowCompressionFactorPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : HikHandle = None
	pass
class RightElbowMaxExtensionPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : HikHandle = None
	pass
class RightKneeCompressionFactorPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : HikHandle = None
	pass
class RightKneeMaxExtensionPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : HikHandle = None
	pass
class RightLegStiffnessPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : HikHandle = None
	pass
class RightShoulderStiffnessPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : HikHandle = None
	pass
class SpineStiffnessPlug(Plug):
	parent : StiffnessPlug = PlugDescriptor("stiffness")
	node : HikHandle = None
	pass
class StiffnessPlug(Plug):
	chestStiffness_ : ChestStiffnessPlug = PlugDescriptor("chestStiffness")
	rco_ : ChestStiffnessPlug = PlugDescriptor("chestStiffness")
	hipsEnforceGravity_ : HipsEnforceGravityPlug = PlugDescriptor("hipsEnforceGravity")
	egr_ : HipsEnforceGravityPlug = PlugDescriptor("hipsEnforceGravity")
	hipsStiffness_ : HipsStiffnessPlug = PlugDescriptor("hipsStiffness")
	rho_ : HipsStiffnessPlug = PlugDescriptor("hipsStiffness")
	leftArmStiffness_ : LeftArmStiffnessPlug = PlugDescriptor("leftArmStiffness")
	rle_ : LeftArmStiffnessPlug = PlugDescriptor("leftArmStiffness")
	leftElbowCompressionFactor_ : LeftElbowCompressionFactorPlug = PlugDescriptor("leftElbowCompressionFactor")
	cle_ : LeftElbowCompressionFactorPlug = PlugDescriptor("leftElbowCompressionFactor")
	leftElbowMaxExtension_ : LeftElbowMaxExtensionPlug = PlugDescriptor("leftElbowMaxExtension")
	mle_ : LeftElbowMaxExtensionPlug = PlugDescriptor("leftElbowMaxExtension")
	leftKneeCompressionFactor_ : LeftKneeCompressionFactorPlug = PlugDescriptor("leftKneeCompressionFactor")
	clk_ : LeftKneeCompressionFactorPlug = PlugDescriptor("leftKneeCompressionFactor")
	leftKneeMaxExtension_ : LeftKneeMaxExtensionPlug = PlugDescriptor("leftKneeMaxExtension")
	mlk_ : LeftKneeMaxExtensionPlug = PlugDescriptor("leftKneeMaxExtension")
	leftLegStiffness_ : LeftLegStiffnessPlug = PlugDescriptor("leftLegStiffness")
	rlk_ : LeftLegStiffnessPlug = PlugDescriptor("leftLegStiffness")
	leftShoulderStiffness_ : LeftShoulderStiffnessPlug = PlugDescriptor("leftShoulderStiffness")
	rlco_ : LeftShoulderStiffnessPlug = PlugDescriptor("leftShoulderStiffness")
	neckStiffness_ : NeckStiffnessPlug = PlugDescriptor("neckStiffness")
	nst_ : NeckStiffnessPlug = PlugDescriptor("neckStiffness")
	rightArmStiffness_ : RightArmStiffnessPlug = PlugDescriptor("rightArmStiffness")
	rre_ : RightArmStiffnessPlug = PlugDescriptor("rightArmStiffness")
	rightElbowCompressionFactor_ : RightElbowCompressionFactorPlug = PlugDescriptor("rightElbowCompressionFactor")
	cre_ : RightElbowCompressionFactorPlug = PlugDescriptor("rightElbowCompressionFactor")
	rightElbowMaxExtension_ : RightElbowMaxExtensionPlug = PlugDescriptor("rightElbowMaxExtension")
	mre_ : RightElbowMaxExtensionPlug = PlugDescriptor("rightElbowMaxExtension")
	rightKneeCompressionFactor_ : RightKneeCompressionFactorPlug = PlugDescriptor("rightKneeCompressionFactor")
	crk_ : RightKneeCompressionFactorPlug = PlugDescriptor("rightKneeCompressionFactor")
	rightKneeMaxExtension_ : RightKneeMaxExtensionPlug = PlugDescriptor("rightKneeMaxExtension")
	mrk_ : RightKneeMaxExtensionPlug = PlugDescriptor("rightKneeMaxExtension")
	rightLegStiffness_ : RightLegStiffnessPlug = PlugDescriptor("rightLegStiffness")
	rrk_ : RightLegStiffnessPlug = PlugDescriptor("rightLegStiffness")
	rightShoulderStiffness_ : RightShoulderStiffnessPlug = PlugDescriptor("rightShoulderStiffness")
	rrc_ : RightShoulderStiffnessPlug = PlugDescriptor("rightShoulderStiffness")
	spineStiffness_ : SpineStiffnessPlug = PlugDescriptor("spineStiffness")
	sst_ : SpineStiffnessPlug = PlugDescriptor("spineStiffness")
	node : HikHandle = None
	pass
class TimePlug(Plug):
	node : HikHandle = None
	pass
class LeftFootExtraFingerTipPlug(Plug):
	parent : ToeTipsSizesPlug = PlugDescriptor("toeTipsSizes")
	node : HikHandle = None
	pass
class LeftFootIndexTipPlug(Plug):
	parent : ToeTipsSizesPlug = PlugDescriptor("toeTipsSizes")
	node : HikHandle = None
	pass
class LeftFootMiddleTipPlug(Plug):
	parent : ToeTipsSizesPlug = PlugDescriptor("toeTipsSizes")
	node : HikHandle = None
	pass
class LeftFootPinkyTipPlug(Plug):
	parent : ToeTipsSizesPlug = PlugDescriptor("toeTipsSizes")
	node : HikHandle = None
	pass
class LeftFootRingTipPlug(Plug):
	parent : ToeTipsSizesPlug = PlugDescriptor("toeTipsSizes")
	node : HikHandle = None
	pass
class LeftFootThumbTipPlug(Plug):
	parent : ToeTipsSizesPlug = PlugDescriptor("toeTipsSizes")
	node : HikHandle = None
	pass
class RightFootExtraFingerTipPlug(Plug):
	parent : ToeTipsSizesPlug = PlugDescriptor("toeTipsSizes")
	node : HikHandle = None
	pass
class RightFootIndexTipPlug(Plug):
	parent : ToeTipsSizesPlug = PlugDescriptor("toeTipsSizes")
	node : HikHandle = None
	pass
class RightFootMiddleTipPlug(Plug):
	parent : ToeTipsSizesPlug = PlugDescriptor("toeTipsSizes")
	node : HikHandle = None
	pass
class RightFootPinkyTipPlug(Plug):
	parent : ToeTipsSizesPlug = PlugDescriptor("toeTipsSizes")
	node : HikHandle = None
	pass
class RightFootRingTipPlug(Plug):
	parent : ToeTipsSizesPlug = PlugDescriptor("toeTipsSizes")
	node : HikHandle = None
	pass
class RightFootThumbTipPlug(Plug):
	parent : ToeTipsSizesPlug = PlugDescriptor("toeTipsSizes")
	node : HikHandle = None
	pass
class ToeTipsSizesPlug(Plug):
	leftFootExtraFingerTip_ : LeftFootExtraFingerTipPlug = PlugDescriptor("leftFootExtraFingerTip")
	txl_ : LeftFootExtraFingerTipPlug = PlugDescriptor("leftFootExtraFingerTip")
	leftFootIndexTip_ : LeftFootIndexTipPlug = PlugDescriptor("leftFootIndexTip")
	til_ : LeftFootIndexTipPlug = PlugDescriptor("leftFootIndexTip")
	leftFootMiddleTip_ : LeftFootMiddleTipPlug = PlugDescriptor("leftFootMiddleTip")
	tml_ : LeftFootMiddleTipPlug = PlugDescriptor("leftFootMiddleTip")
	leftFootPinkyTip_ : LeftFootPinkyTipPlug = PlugDescriptor("leftFootPinkyTip")
	tpl_ : LeftFootPinkyTipPlug = PlugDescriptor("leftFootPinkyTip")
	leftFootRingTip_ : LeftFootRingTipPlug = PlugDescriptor("leftFootRingTip")
	trl_ : LeftFootRingTipPlug = PlugDescriptor("leftFootRingTip")
	leftFootThumbTip_ : LeftFootThumbTipPlug = PlugDescriptor("leftFootThumbTip")
	ttl_ : LeftFootThumbTipPlug = PlugDescriptor("leftFootThumbTip")
	rightFootExtraFingerTip_ : RightFootExtraFingerTipPlug = PlugDescriptor("rightFootExtraFingerTip")
	txr_ : RightFootExtraFingerTipPlug = PlugDescriptor("rightFootExtraFingerTip")
	rightFootIndexTip_ : RightFootIndexTipPlug = PlugDescriptor("rightFootIndexTip")
	tir_ : RightFootIndexTipPlug = PlugDescriptor("rightFootIndexTip")
	rightFootMiddleTip_ : RightFootMiddleTipPlug = PlugDescriptor("rightFootMiddleTip")
	tmr_ : RightFootMiddleTipPlug = PlugDescriptor("rightFootMiddleTip")
	rightFootPinkyTip_ : RightFootPinkyTipPlug = PlugDescriptor("rightFootPinkyTip")
	tpr_ : RightFootPinkyTipPlug = PlugDescriptor("rightFootPinkyTip")
	rightFootRingTip_ : RightFootRingTipPlug = PlugDescriptor("rightFootRingTip")
	trr_ : RightFootRingTipPlug = PlugDescriptor("rightFootRingTip")
	rightFootThumbTip_ : RightFootThumbTipPlug = PlugDescriptor("rightFootThumbTip")
	ttr_ : RightFootThumbTipPlug = PlugDescriptor("rightFootThumbTip")
	node : HikHandle = None
	pass
class ToesContactRollStiffnessPlug(Plug):
	parent : ToesFloorContactSetupPlug = PlugDescriptor("toesFloorContactSetup")
	node : HikHandle = None
	pass
class ToesContactTypePlug(Plug):
	parent : ToesFloorContactSetupPlug = PlugDescriptor("toesFloorContactSetup")
	node : HikHandle = None
	pass
class ToesFloorContactSetupPlug(Plug):
	toesContactRollStiffness_ : ToesContactRollStiffnessPlug = PlugDescriptor("toesContactRollStiffness")
	fcr_ : ToesContactRollStiffnessPlug = PlugDescriptor("toesContactRollStiffness")
	toesContactType_ : ToesContactTypePlug = PlugDescriptor("toesContactType")
	tct_ : ToesContactTypePlug = PlugDescriptor("toesContactType")
	node : HikHandle = None
	pass
class UsingMB55RigPlug(Plug):
	node : HikHandle = None
	pass
# endregion


# define node class
class HikHandle(IkHandle):
	activate_ : ActivatePlug = PlugDescriptor("activate")
	chestPull_ : ChestPullPlug = PlugDescriptor("chestPull")
	chest_ : ChestPlug = PlugDescriptor("chest")
	handBack_ : HandBackPlug = PlugDescriptor("handBack")
	handFront_ : HandFrontPlug = PlugDescriptor("handFront")
	handHeight_ : HandHeightPlug = PlugDescriptor("handHeight")
	handInSide_ : HandInSidePlug = PlugDescriptor("handInSide")
	handMiddle_ : HandMiddlePlug = PlugDescriptor("handMiddle")
	handOutSide_ : HandOutSidePlug = PlugDescriptor("handOutSide")
	contactsPosition_ : ContactsPositionPlug = PlugDescriptor("contactsPosition")
	convertScale_ : ConvertScalePlug = PlugDescriptor("convertScale")
	defaultMatrix_ : DefaultMatrixPlug = PlugDescriptor("defaultMatrix")
	effectors_ : EffectorsPlug = PlugDescriptor("effectors")
	pullIterationCount_ : PullIterationCountPlug = PlugDescriptor("pullIterationCount")
	extra_ : ExtraPlug = PlugDescriptor("extra")
	footBack_ : FootBackPlug = PlugDescriptor("footBack")
	footFront_ : FootFrontPlug = PlugDescriptor("footFront")
	footHeight_ : FootHeightPlug = PlugDescriptor("footHeight")
	footInSide_ : FootInSidePlug = PlugDescriptor("footInSide")
	footMiddle_ : FootMiddlePlug = PlugDescriptor("footMiddle")
	footOutSide_ : FootOutSidePlug = PlugDescriptor("footOutSide")
	feetContactPosition_ : FeetContactPositionPlug = PlugDescriptor("feetContactPosition")
	feetContactStiffness_ : FeetContactStiffnessPlug = PlugDescriptor("feetContactStiffness")
	feetContactType_ : FeetContactTypePlug = PlugDescriptor("feetContactType")
	feetFloorPivot_ : FeetFloorPivotPlug = PlugDescriptor("feetFloorPivot")
	feetFloorContactSetup_ : FeetFloorContactSetupPlug = PlugDescriptor("feetFloorContactSetup")
	leftHandExtraFingerTip_ : LeftHandExtraFingerTipPlug = PlugDescriptor("leftHandExtraFingerTip")
	leftHandIndexTip_ : LeftHandIndexTipPlug = PlugDescriptor("leftHandIndexTip")
	leftHandMiddleTip_ : LeftHandMiddleTipPlug = PlugDescriptor("leftHandMiddleTip")
	leftHandPinkyTip_ : LeftHandPinkyTipPlug = PlugDescriptor("leftHandPinkyTip")
	leftHandRingTip_ : LeftHandRingTipPlug = PlugDescriptor("leftHandRingTip")
	leftHandThumbTip_ : LeftHandThumbTipPlug = PlugDescriptor("leftHandThumbTip")
	rightHandExtraFingerTip_ : RightHandExtraFingerTipPlug = PlugDescriptor("rightHandExtraFingerTip")
	rightHandIndexTip_ : RightHandIndexTipPlug = PlugDescriptor("rightHandIndexTip")
	rightHandMiddleTip_ : RightHandMiddleTipPlug = PlugDescriptor("rightHandMiddleTip")
	rightHandPinkyTip_ : RightHandPinkyTipPlug = PlugDescriptor("rightHandPinkyTip")
	rightHandRingTip_ : RightHandRingTipPlug = PlugDescriptor("rightHandRingTip")
	rightHandThumbTip_ : RightHandThumbTipPlug = PlugDescriptor("rightHandThumbTip")
	fingerTipsSizes_ : FingerTipsSizesPlug = PlugDescriptor("fingerTipsSizes")
	fingersContactRollStiffness_ : FingersContactRollStiffnessPlug = PlugDescriptor("fingersContactRollStiffness")
	fingersContactType_ : FingersContactTypePlug = PlugDescriptor("fingersContactType")
	fingersFloorContactSetup_ : FingersFloorContactSetupPlug = PlugDescriptor("fingersFloorContactSetup")
	fkjoints_ : FkjointsPlug = PlugDescriptor("fkjoints")
	fkmatrix_ : FkmatrixPlug = PlugDescriptor("fkmatrix")
	feetFloorContact_ : FeetFloorContactPlug = PlugDescriptor("feetFloorContact")
	fingersFloorContact_ : FingersFloorContactPlug = PlugDescriptor("fingersFloorContact")
	handsFloorContact_ : HandsFloorContactPlug = PlugDescriptor("handsFloorContact")
	toesFloorContact_ : ToesFloorContactPlug = PlugDescriptor("toesFloorContact")
	floorContacts_ : FloorContactsPlug = PlugDescriptor("floorContacts")
	handsContactStiffness_ : HandsContactStiffnessPlug = PlugDescriptor("handsContactStiffness")
	handsContactType_ : HandsContactTypePlug = PlugDescriptor("handsContactType")
	handsFloorPivot_ : HandsFloorPivotPlug = PlugDescriptor("handsFloorPivot")
	handsFloorContactSetup_ : HandsFloorContactSetupPlug = PlugDescriptor("handsFloorContactSetup")
	headPull_ : HeadPullPlug = PlugDescriptor("headPull")
	head_ : HeadPlug = PlugDescriptor("head")
	hipsPull_ : HipsPullPlug = PlugDescriptor("hipsPull")
	hips_ : HipsPlug = PlugDescriptor("hips")
	joints_ : JointsPlug = PlugDescriptor("joints")
	leftElbowKillPitch_ : LeftElbowKillPitchPlug = PlugDescriptor("leftElbowKillPitch")
	leftKneeKillPitch_ : LeftKneeKillPitchPlug = PlugDescriptor("leftKneeKillPitch")
	rightElbowKillPitch_ : RightElbowKillPitchPlug = PlugDescriptor("rightElbowKillPitch")
	rightKneeKillPitch_ : RightKneeKillPitchPlug = PlugDescriptor("rightKneeKillPitch")
	killPitch_ : KillPitchPlug = PlugDescriptor("killPitch")
	leftElbowPull_ : LeftElbowPullPlug = PlugDescriptor("leftElbowPull")
	leftFingerBasePull_ : LeftFingerBasePullPlug = PlugDescriptor("leftFingerBasePull")
	leftHandPullChest_ : LeftHandPullChestPlug = PlugDescriptor("leftHandPullChest")
	leftHandPullHips_ : LeftHandPullHipsPlug = PlugDescriptor("leftHandPullHips")
	leftArm_ : LeftArmPlug = PlugDescriptor("leftArm")
	leftFootGroundPlane_ : LeftFootGroundPlanePlug = PlugDescriptor("leftFootGroundPlane")
	leftFootOrientedGroundPlane_ : LeftFootOrientedGroundPlanePlug = PlugDescriptor("leftFootOrientedGroundPlane")
	leftHandGroundPlane_ : LeftHandGroundPlanePlug = PlugDescriptor("leftHandGroundPlane")
	leftHandOrientedGroundPlane_ : LeftHandOrientedGroundPlanePlug = PlugDescriptor("leftHandOrientedGroundPlane")
	leftFootPull_ : LeftFootPullPlug = PlugDescriptor("leftFootPull")
	leftKneePull_ : LeftKneePullPlug = PlugDescriptor("leftKneePull")
	leftToeBasePull_ : LeftToeBasePullPlug = PlugDescriptor("leftToeBasePull")
	leftLeg_ : LeftLegPlug = PlugDescriptor("leftLeg")
	propertyChanged_ : PropertyChangedPlug = PlugDescriptor("propertyChanged")
	rightElbowPull_ : RightElbowPullPlug = PlugDescriptor("rightElbowPull")
	rightFingerBasePull_ : RightFingerBasePullPlug = PlugDescriptor("rightFingerBasePull")
	rightHandPullChest_ : RightHandPullChestPlug = PlugDescriptor("rightHandPullChest")
	rightHandPullHips_ : RightHandPullHipsPlug = PlugDescriptor("rightHandPullHips")
	rightArm_ : RightArmPlug = PlugDescriptor("rightArm")
	rightFootGroundPlane_ : RightFootGroundPlanePlug = PlugDescriptor("rightFootGroundPlane")
	rightFootOrientedGroundPlane_ : RightFootOrientedGroundPlanePlug = PlugDescriptor("rightFootOrientedGroundPlane")
	rightHandGroundPlane_ : RightHandGroundPlanePlug = PlugDescriptor("rightHandGroundPlane")
	rightHandOrientedGroundPlane_ : RightHandOrientedGroundPlanePlug = PlugDescriptor("rightHandOrientedGroundPlane")
	rightFootPull_ : RightFootPullPlug = PlugDescriptor("rightFootPull")
	rightKneePull_ : RightKneePullPlug = PlugDescriptor("rightKneePull")
	rightToeBasePull_ : RightToeBasePullPlug = PlugDescriptor("rightToeBasePull")
	rightLeg_ : RightLegPlug = PlugDescriptor("rightLeg")
	leftArmRoll_ : LeftArmRollPlug = PlugDescriptor("leftArmRoll")
	leftArmRollMode_ : LeftArmRollModePlug = PlugDescriptor("leftArmRollMode")
	leftForeArmRoll_ : LeftForeArmRollPlug = PlugDescriptor("leftForeArmRoll")
	leftForeArmRollMode_ : LeftForeArmRollModePlug = PlugDescriptor("leftForeArmRollMode")
	leftLegRoll_ : LeftLegRollPlug = PlugDescriptor("leftLegRoll")
	leftLegRollMode_ : LeftLegRollModePlug = PlugDescriptor("leftLegRollMode")
	leftUpLegRoll_ : LeftUpLegRollPlug = PlugDescriptor("leftUpLegRoll")
	leftUpLegRollMode_ : LeftUpLegRollModePlug = PlugDescriptor("leftUpLegRollMode")
	rightArmRoll_ : RightArmRollPlug = PlugDescriptor("rightArmRoll")
	rightArmRollMode_ : RightArmRollModePlug = PlugDescriptor("rightArmRollMode")
	rightForeArmRoll_ : RightForeArmRollPlug = PlugDescriptor("rightForeArmRoll")
	rightForeArmRollMode_ : RightForeArmRollModePlug = PlugDescriptor("rightForeArmRollMode")
	rightLegRoll_ : RightLegRollPlug = PlugDescriptor("rightLegRoll")
	rightLegRollMode_ : RightLegRollModePlug = PlugDescriptor("rightLegRollMode")
	rightUpLegRoll_ : RightUpLegRollPlug = PlugDescriptor("rightUpLegRoll")
	rightUpLegRollMode_ : RightUpLegRollModePlug = PlugDescriptor("rightUpLegRollMode")
	rollExtractionMode_ : RollExtractionModePlug = PlugDescriptor("rollExtractionMode")
	rollExtraction_ : RollExtractionPlug = PlugDescriptor("rollExtraction")
	expertMode_ : ExpertModePlug = PlugDescriptor("expertMode")
	hipTranslationMode_ : HipTranslationModePlug = PlugDescriptor("hipTranslationMode")
	postureType_ : PostureTypePlug = PlugDescriptor("postureType")
	realisticShoulderSolving_ : RealisticShoulderSolvingPlug = PlugDescriptor("realisticShoulderSolving")
	solveFingers_ : SolveFingersPlug = PlugDescriptor("solveFingers")
	solving_ : SolvingPlug = PlugDescriptor("solving")
	stancePoseMatrix_ : StancePoseMatrixPlug = PlugDescriptor("stancePoseMatrix")
	chestStiffness_ : ChestStiffnessPlug = PlugDescriptor("chestStiffness")
	hipsEnforceGravity_ : HipsEnforceGravityPlug = PlugDescriptor("hipsEnforceGravity")
	hipsStiffness_ : HipsStiffnessPlug = PlugDescriptor("hipsStiffness")
	leftArmStiffness_ : LeftArmStiffnessPlug = PlugDescriptor("leftArmStiffness")
	leftElbowCompressionFactor_ : LeftElbowCompressionFactorPlug = PlugDescriptor("leftElbowCompressionFactor")
	leftElbowMaxExtension_ : LeftElbowMaxExtensionPlug = PlugDescriptor("leftElbowMaxExtension")
	leftKneeCompressionFactor_ : LeftKneeCompressionFactorPlug = PlugDescriptor("leftKneeCompressionFactor")
	leftKneeMaxExtension_ : LeftKneeMaxExtensionPlug = PlugDescriptor("leftKneeMaxExtension")
	leftLegStiffness_ : LeftLegStiffnessPlug = PlugDescriptor("leftLegStiffness")
	leftShoulderStiffness_ : LeftShoulderStiffnessPlug = PlugDescriptor("leftShoulderStiffness")
	neckStiffness_ : NeckStiffnessPlug = PlugDescriptor("neckStiffness")
	rightArmStiffness_ : RightArmStiffnessPlug = PlugDescriptor("rightArmStiffness")
	rightElbowCompressionFactor_ : RightElbowCompressionFactorPlug = PlugDescriptor("rightElbowCompressionFactor")
	rightElbowMaxExtension_ : RightElbowMaxExtensionPlug = PlugDescriptor("rightElbowMaxExtension")
	rightKneeCompressionFactor_ : RightKneeCompressionFactorPlug = PlugDescriptor("rightKneeCompressionFactor")
	rightKneeMaxExtension_ : RightKneeMaxExtensionPlug = PlugDescriptor("rightKneeMaxExtension")
	rightLegStiffness_ : RightLegStiffnessPlug = PlugDescriptor("rightLegStiffness")
	rightShoulderStiffness_ : RightShoulderStiffnessPlug = PlugDescriptor("rightShoulderStiffness")
	spineStiffness_ : SpineStiffnessPlug = PlugDescriptor("spineStiffness")
	stiffness_ : StiffnessPlug = PlugDescriptor("stiffness")
	time_ : TimePlug = PlugDescriptor("time")
	leftFootExtraFingerTip_ : LeftFootExtraFingerTipPlug = PlugDescriptor("leftFootExtraFingerTip")
	leftFootIndexTip_ : LeftFootIndexTipPlug = PlugDescriptor("leftFootIndexTip")
	leftFootMiddleTip_ : LeftFootMiddleTipPlug = PlugDescriptor("leftFootMiddleTip")
	leftFootPinkyTip_ : LeftFootPinkyTipPlug = PlugDescriptor("leftFootPinkyTip")
	leftFootRingTip_ : LeftFootRingTipPlug = PlugDescriptor("leftFootRingTip")
	leftFootThumbTip_ : LeftFootThumbTipPlug = PlugDescriptor("leftFootThumbTip")
	rightFootExtraFingerTip_ : RightFootExtraFingerTipPlug = PlugDescriptor("rightFootExtraFingerTip")
	rightFootIndexTip_ : RightFootIndexTipPlug = PlugDescriptor("rightFootIndexTip")
	rightFootMiddleTip_ : RightFootMiddleTipPlug = PlugDescriptor("rightFootMiddleTip")
	rightFootPinkyTip_ : RightFootPinkyTipPlug = PlugDescriptor("rightFootPinkyTip")
	rightFootRingTip_ : RightFootRingTipPlug = PlugDescriptor("rightFootRingTip")
	rightFootThumbTip_ : RightFootThumbTipPlug = PlugDescriptor("rightFootThumbTip")
	toeTipsSizes_ : ToeTipsSizesPlug = PlugDescriptor("toeTipsSizes")
	toesContactRollStiffness_ : ToesContactRollStiffnessPlug = PlugDescriptor("toesContactRollStiffness")
	toesContactType_ : ToesContactTypePlug = PlugDescriptor("toesContactType")
	toesFloorContactSetup_ : ToesFloorContactSetupPlug = PlugDescriptor("toesFloorContactSetup")
	usingMB55Rig_ : UsingMB55RigPlug = PlugDescriptor("usingMB55Rig")

	# node attributes

	typeName = "hikHandle"
	apiTypeInt = 965
	apiTypeStr = "kHikHandle"
	typeIdInt = 1263028552
	MFnCls = om.MFnTransform
	nodeLeafClassAttrs = ["activate", "chestPull", "chest", "handBack", "handFront", "handHeight", "handInSide", "handMiddle", "handOutSide", "contactsPosition", "convertScale", "defaultMatrix", "effectors", "pullIterationCount", "extra", "footBack", "footFront", "footHeight", "footInSide", "footMiddle", "footOutSide", "feetContactPosition", "feetContactStiffness", "feetContactType", "feetFloorPivot", "feetFloorContactSetup", "leftHandExtraFingerTip", "leftHandIndexTip", "leftHandMiddleTip", "leftHandPinkyTip", "leftHandRingTip", "leftHandThumbTip", "rightHandExtraFingerTip", "rightHandIndexTip", "rightHandMiddleTip", "rightHandPinkyTip", "rightHandRingTip", "rightHandThumbTip", "fingerTipsSizes", "fingersContactRollStiffness", "fingersContactType", "fingersFloorContactSetup", "fkjoints", "fkmatrix", "feetFloorContact", "fingersFloorContact", "handsFloorContact", "toesFloorContact", "floorContacts", "handsContactStiffness", "handsContactType", "handsFloorPivot", "handsFloorContactSetup", "headPull", "head", "hipsPull", "hips", "joints", "leftElbowKillPitch", "leftKneeKillPitch", "rightElbowKillPitch", "rightKneeKillPitch", "killPitch", "leftElbowPull", "leftFingerBasePull", "leftHandPullChest", "leftHandPullHips", "leftArm", "leftFootGroundPlane", "leftFootOrientedGroundPlane", "leftHandGroundPlane", "leftHandOrientedGroundPlane", "leftFootPull", "leftKneePull", "leftToeBasePull", "leftLeg", "propertyChanged", "rightElbowPull", "rightFingerBasePull", "rightHandPullChest", "rightHandPullHips", "rightArm", "rightFootGroundPlane", "rightFootOrientedGroundPlane", "rightHandGroundPlane", "rightHandOrientedGroundPlane", "rightFootPull", "rightKneePull", "rightToeBasePull", "rightLeg", "leftArmRoll", "leftArmRollMode", "leftForeArmRoll", "leftForeArmRollMode", "leftLegRoll", "leftLegRollMode", "leftUpLegRoll", "leftUpLegRollMode", "rightArmRoll", "rightArmRollMode", "rightForeArmRoll", "rightForeArmRollMode", "rightLegRoll", "rightLegRollMode", "rightUpLegRoll", "rightUpLegRollMode", "rollExtractionMode", "rollExtraction", "expertMode", "hipTranslationMode", "postureType", "realisticShoulderSolving", "solveFingers", "solving", "stancePoseMatrix", "chestStiffness", "hipsEnforceGravity", "hipsStiffness", "leftArmStiffness", "leftElbowCompressionFactor", "leftElbowMaxExtension", "leftKneeCompressionFactor", "leftKneeMaxExtension", "leftLegStiffness", "leftShoulderStiffness", "neckStiffness", "rightArmStiffness", "rightElbowCompressionFactor", "rightElbowMaxExtension", "rightKneeCompressionFactor", "rightKneeMaxExtension", "rightLegStiffness", "rightShoulderStiffness", "spineStiffness", "stiffness", "time", "leftFootExtraFingerTip", "leftFootIndexTip", "leftFootMiddleTip", "leftFootPinkyTip", "leftFootRingTip", "leftFootThumbTip", "rightFootExtraFingerTip", "rightFootIndexTip", "rightFootMiddleTip", "rightFootPinkyTip", "rightFootRingTip", "rightFootThumbTip", "toeTipsSizes", "toesContactRollStiffness", "toesContactType", "toesFloorContactSetup", "usingMB55Rig"]
	nodeLeafPlugs = ["activate", "chest", "contactsPosition", "convertScale", "defaultMatrix", "effectors", "extra", "feetContactPosition", "feetFloorContactSetup", "fingerTipsSizes", "fingersFloorContactSetup", "fkjoints", "fkmatrix", "floorContacts", "handsFloorContactSetup", "head", "hips", "joints", "killPitch", "leftArm", "leftFootGroundPlane", "leftFootOrientedGroundPlane", "leftHandGroundPlane", "leftHandOrientedGroundPlane", "leftLeg", "propertyChanged", "rightArm", "rightFootGroundPlane", "rightFootOrientedGroundPlane", "rightHandGroundPlane", "rightHandOrientedGroundPlane", "rightLeg", "rollExtraction", "solving", "stancePoseMatrix", "stiffness", "time", "toeTipsSizes", "toesFloorContactSetup", "usingMB55Rig"]
	pass

