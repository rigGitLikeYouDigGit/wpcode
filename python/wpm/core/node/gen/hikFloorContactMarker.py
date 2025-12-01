

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Locator = retriever.getNodeCls("Locator")
assert Locator
if T.TYPE_CHECKING:
	from .. import Locator

# add node doc



# region plug type defs
class HandBackPlug(Plug):
	parent : ContactsPositionPlug = PlugDescriptor("contactsPosition")
	node : HikFloorContactMarker = None
	pass
class HandFrontPlug(Plug):
	parent : ContactsPositionPlug = PlugDescriptor("contactsPosition")
	node : HikFloorContactMarker = None
	pass
class HandHeightPlug(Plug):
	parent : ContactsPositionPlug = PlugDescriptor("contactsPosition")
	node : HikFloorContactMarker = None
	pass
class HandInSidePlug(Plug):
	parent : ContactsPositionPlug = PlugDescriptor("contactsPosition")
	node : HikFloorContactMarker = None
	pass
class HandMiddlePlug(Plug):
	parent : ContactsPositionPlug = PlugDescriptor("contactsPosition")
	node : HikFloorContactMarker = None
	pass
class HandOutSidePlug(Plug):
	parent : ContactsPositionPlug = PlugDescriptor("contactsPosition")
	node : HikFloorContactMarker = None
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
	node : HikFloorContactMarker = None
	pass
class FootBackPlug(Plug):
	parent : FeetContactPositionPlug = PlugDescriptor("feetContactPosition")
	node : HikFloorContactMarker = None
	pass
class FootFrontPlug(Plug):
	parent : FeetContactPositionPlug = PlugDescriptor("feetContactPosition")
	node : HikFloorContactMarker = None
	pass
class FootHeightPlug(Plug):
	parent : FeetContactPositionPlug = PlugDescriptor("feetContactPosition")
	node : HikFloorContactMarker = None
	pass
class FootInSidePlug(Plug):
	parent : FeetContactPositionPlug = PlugDescriptor("feetContactPosition")
	node : HikFloorContactMarker = None
	pass
class FootMiddlePlug(Plug):
	parent : FeetContactPositionPlug = PlugDescriptor("feetContactPosition")
	node : HikFloorContactMarker = None
	pass
class FootOutSidePlug(Plug):
	parent : FeetContactPositionPlug = PlugDescriptor("feetContactPosition")
	node : HikFloorContactMarker = None
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
	node : HikFloorContactMarker = None
	pass
class FeetContactStiffnessPlug(Plug):
	parent : FeetFloorContactSetupPlug = PlugDescriptor("feetFloorContactSetup")
	node : HikFloorContactMarker = None
	pass
class FeetContactTypePlug(Plug):
	parent : FeetFloorContactSetupPlug = PlugDescriptor("feetFloorContactSetup")
	node : HikFloorContactMarker = None
	pass
class FeetFloorPivotPlug(Plug):
	parent : FeetFloorContactSetupPlug = PlugDescriptor("feetFloorContactSetup")
	node : HikFloorContactMarker = None
	pass
class FeetFloorContactSetupPlug(Plug):
	feetContactStiffness_ : FeetContactStiffnessPlug = PlugDescriptor("feetContactStiffness")
	fcs_ : FeetContactStiffnessPlug = PlugDescriptor("feetContactStiffness")
	feetContactType_ : FeetContactTypePlug = PlugDescriptor("feetContactType")
	fct_ : FeetContactTypePlug = PlugDescriptor("feetContactType")
	feetFloorPivot_ : FeetFloorPivotPlug = PlugDescriptor("feetFloorPivot")
	fpv_ : FeetFloorPivotPlug = PlugDescriptor("feetFloorPivot")
	node : HikFloorContactMarker = None
	pass
class FingersContactRollStiffnessPlug(Plug):
	parent : FingersFloorContactSetupPlug = PlugDescriptor("fingersFloorContactSetup")
	node : HikFloorContactMarker = None
	pass
class FingersContactTypePlug(Plug):
	parent : FingersFloorContactSetupPlug = PlugDescriptor("fingersFloorContactSetup")
	node : HikFloorContactMarker = None
	pass
class FingersFloorContactSetupPlug(Plug):
	fingersContactRollStiffness_ : FingersContactRollStiffnessPlug = PlugDescriptor("fingersContactRollStiffness")
	hcr_ : FingersContactRollStiffnessPlug = PlugDescriptor("fingersContactRollStiffness")
	fingersContactType_ : FingersContactTypePlug = PlugDescriptor("fingersContactType")
	fcm_ : FingersContactTypePlug = PlugDescriptor("fingersContactType")
	node : HikFloorContactMarker = None
	pass
class DrawFeetContactPlug(Plug):
	parent : FloorContactsPlug = PlugDescriptor("floorContacts")
	node : HikFloorContactMarker = None
	pass
class DrawHandContactPlug(Plug):
	parent : FloorContactsPlug = PlugDescriptor("floorContacts")
	node : HikFloorContactMarker = None
	pass
class FeetContactPlug(Plug):
	parent : FloorContactsPlug = PlugDescriptor("floorContacts")
	node : HikFloorContactMarker = None
	pass
class FingersContactPlug(Plug):
	parent : FloorContactsPlug = PlugDescriptor("floorContacts")
	node : HikFloorContactMarker = None
	pass
class HandsContactPlug(Plug):
	parent : FloorContactsPlug = PlugDescriptor("floorContacts")
	node : HikFloorContactMarker = None
	pass
class ToesContactPlug(Plug):
	parent : FloorContactsPlug = PlugDescriptor("floorContacts")
	node : HikFloorContactMarker = None
	pass
class FloorContactsPlug(Plug):
	drawFeetContact_ : DrawFeetContactPlug = PlugDescriptor("drawFeetContact")
	dfc_ : DrawFeetContactPlug = PlugDescriptor("drawFeetContact")
	drawHandContact_ : DrawHandContactPlug = PlugDescriptor("drawHandContact")
	dhc_ : DrawHandContactPlug = PlugDescriptor("drawHandContact")
	feetContact_ : FeetContactPlug = PlugDescriptor("feetContact")
	fec_ : FeetContactPlug = PlugDescriptor("feetContact")
	fingersContact_ : FingersContactPlug = PlugDescriptor("fingersContact")
	fic_ : FingersContactPlug = PlugDescriptor("fingersContact")
	handsContact_ : HandsContactPlug = PlugDescriptor("handsContact")
	hfc_ : HandsContactPlug = PlugDescriptor("handsContact")
	toesContact_ : ToesContactPlug = PlugDescriptor("toesContact")
	tfc_ : ToesContactPlug = PlugDescriptor("toesContact")
	node : HikFloorContactMarker = None
	pass
class HandsContactStiffnessPlug(Plug):
	parent : HandsFloorContactSetupPlug = PlugDescriptor("handsFloorContactSetup")
	node : HikFloorContactMarker = None
	pass
class HandsContactTypePlug(Plug):
	parent : HandsFloorContactSetupPlug = PlugDescriptor("handsFloorContactSetup")
	node : HikFloorContactMarker = None
	pass
class HandsFloorPivotPlug(Plug):
	parent : HandsFloorContactSetupPlug = PlugDescriptor("handsFloorContactSetup")
	node : HikFloorContactMarker = None
	pass
class HandsFloorContactSetupPlug(Plug):
	handsContactStiffness_ : HandsContactStiffnessPlug = PlugDescriptor("handsContactStiffness")
	hcs_ : HandsContactStiffnessPlug = PlugDescriptor("handsContactStiffness")
	handsContactType_ : HandsContactTypePlug = PlugDescriptor("handsContactType")
	hct_ : HandsContactTypePlug = PlugDescriptor("handsContactType")
	handsFloorPivot_ : HandsFloorPivotPlug = PlugDescriptor("handsFloorPivot")
	hfp_ : HandsFloorPivotPlug = PlugDescriptor("handsFloorPivot")
	node : HikFloorContactMarker = None
	pass
class MarkerSizePlug(Plug):
	node : HikFloorContactMarker = None
	pass
class ToesContactRollStiffnessPlug(Plug):
	parent : ToesFloorContactSetupPlug = PlugDescriptor("toesFloorContactSetup")
	node : HikFloorContactMarker = None
	pass
class ToesContactTypePlug(Plug):
	parent : ToesFloorContactSetupPlug = PlugDescriptor("toesFloorContactSetup")
	node : HikFloorContactMarker = None
	pass
class ToesFloorContactSetupPlug(Plug):
	toesContactRollStiffness_ : ToesContactRollStiffnessPlug = PlugDescriptor("toesContactRollStiffness")
	fcr_ : ToesContactRollStiffnessPlug = PlugDescriptor("toesContactRollStiffness")
	toesContactType_ : ToesContactTypePlug = PlugDescriptor("toesContactType")
	tct_ : ToesContactTypePlug = PlugDescriptor("toesContactType")
	node : HikFloorContactMarker = None
	pass
# endregion


# define node class
class HikFloorContactMarker(Locator):
	handBack_ : HandBackPlug = PlugDescriptor("handBack")
	handFront_ : HandFrontPlug = PlugDescriptor("handFront")
	handHeight_ : HandHeightPlug = PlugDescriptor("handHeight")
	handInSide_ : HandInSidePlug = PlugDescriptor("handInSide")
	handMiddle_ : HandMiddlePlug = PlugDescriptor("handMiddle")
	handOutSide_ : HandOutSidePlug = PlugDescriptor("handOutSide")
	contactsPosition_ : ContactsPositionPlug = PlugDescriptor("contactsPosition")
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
	fingersContactRollStiffness_ : FingersContactRollStiffnessPlug = PlugDescriptor("fingersContactRollStiffness")
	fingersContactType_ : FingersContactTypePlug = PlugDescriptor("fingersContactType")
	fingersFloorContactSetup_ : FingersFloorContactSetupPlug = PlugDescriptor("fingersFloorContactSetup")
	drawFeetContact_ : DrawFeetContactPlug = PlugDescriptor("drawFeetContact")
	drawHandContact_ : DrawHandContactPlug = PlugDescriptor("drawHandContact")
	feetContact_ : FeetContactPlug = PlugDescriptor("feetContact")
	fingersContact_ : FingersContactPlug = PlugDescriptor("fingersContact")
	handsContact_ : HandsContactPlug = PlugDescriptor("handsContact")
	toesContact_ : ToesContactPlug = PlugDescriptor("toesContact")
	floorContacts_ : FloorContactsPlug = PlugDescriptor("floorContacts")
	handsContactStiffness_ : HandsContactStiffnessPlug = PlugDescriptor("handsContactStiffness")
	handsContactType_ : HandsContactTypePlug = PlugDescriptor("handsContactType")
	handsFloorPivot_ : HandsFloorPivotPlug = PlugDescriptor("handsFloorPivot")
	handsFloorContactSetup_ : HandsFloorContactSetupPlug = PlugDescriptor("handsFloorContactSetup")
	markerSize_ : MarkerSizePlug = PlugDescriptor("markerSize")
	toesContactRollStiffness_ : ToesContactRollStiffnessPlug = PlugDescriptor("toesContactRollStiffness")
	toesContactType_ : ToesContactTypePlug = PlugDescriptor("toesContactType")
	toesFloorContactSetup_ : ToesFloorContactSetupPlug = PlugDescriptor("toesFloorContactSetup")

	# node attributes

	typeName = "hikFloorContactMarker"
	apiTypeInt = 983
	apiTypeStr = "kHikFloorContactMarker"
	typeIdInt = 1212564301
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["handBack", "handFront", "handHeight", "handInSide", "handMiddle", "handOutSide", "contactsPosition", "footBack", "footFront", "footHeight", "footInSide", "footMiddle", "footOutSide", "feetContactPosition", "feetContactStiffness", "feetContactType", "feetFloorPivot", "feetFloorContactSetup", "fingersContactRollStiffness", "fingersContactType", "fingersFloorContactSetup", "drawFeetContact", "drawHandContact", "feetContact", "fingersContact", "handsContact", "toesContact", "floorContacts", "handsContactStiffness", "handsContactType", "handsFloorPivot", "handsFloorContactSetup", "markerSize", "toesContactRollStiffness", "toesContactType", "toesFloorContactSetup"]
	nodeLeafPlugs = ["contactsPosition", "feetContactPosition", "feetFloorContactSetup", "fingersFloorContactSetup", "floorContacts", "handsFloorContactSetup", "markerSize", "toesFloorContactSetup"]
	pass

