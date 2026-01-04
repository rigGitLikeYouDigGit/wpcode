

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class XCoordPlug(Plug):
	parent : AllCoordsPlug = PlugDescriptor("allCoords")
	node : Flow = None
	pass
class YCoordPlug(Plug):
	parent : AllCoordsPlug = PlugDescriptor("allCoords")
	node : Flow = None
	pass
class ZCoordPlug(Plug):
	parent : AllCoordsPlug = PlugDescriptor("allCoords")
	node : Flow = None
	pass
class AllCoordsPlug(Plug):
	xCoord_ : XCoordPlug = PlugDescriptor("xCoord")
	xc_ : XCoordPlug = PlugDescriptor("xCoord")
	yCoord_ : YCoordPlug = PlugDescriptor("yCoord")
	yc_ : YCoordPlug = PlugDescriptor("yCoord")
	zCoord_ : ZCoordPlug = PlugDescriptor("zCoord")
	zc_ : ZCoordPlug = PlugDescriptor("zCoord")
	node : Flow = None
	pass
class BinMembershipPlug(Plug):
	node : Flow = None
	pass
class CenterXPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : Flow = None
	pass
class CenterYPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : Flow = None
	pass
class CenterZPlug(Plug):
	parent : CenterPlug = PlugDescriptor("center")
	node : Flow = None
	pass
class CenterPlug(Plug):
	centerX_ : CenterXPlug = PlugDescriptor("centerX")
	ctx_ : CenterXPlug = PlugDescriptor("centerX")
	centerY_ : CenterYPlug = PlugDescriptor("centerY")
	cty_ : CenterYPlug = PlugDescriptor("centerY")
	centerZ_ : CenterZPlug = PlugDescriptor("centerZ")
	ctz_ : CenterZPlug = PlugDescriptor("centerZ")
	node : Flow = None
	pass
class CurvePlug(Plug):
	node : Flow = None
	pass
class DefMatrixInvPlug(Plug):
	node : Flow = None
	pass
class DefPtsPlug(Plug):
	node : Flow = None
	pass
class InBaseMatrixPlug(Plug):
	node : Flow = None
	pass
class LatticeOnObjectPlug(Plug):
	node : Flow = None
	pass
class MotionPathPlug(Plug):
	node : Flow = None
	pass
class ObjectWorldMatrixPlug(Plug):
	node : Flow = None
	pass
class OrientMatrixPlug(Plug):
	node : Flow = None
	pass
class OutBaseMatrixPlug(Plug):
	node : Flow = None
	pass
class ParmValuePlug(Plug):
	node : Flow = None
	pass
class SDivisionsPlug(Plug):
	node : Flow = None
	pass
class SetFrontAxisPlug(Plug):
	node : Flow = None
	pass
class SetUpAxisPlug(Plug):
	node : Flow = None
	pass
class TDivisionsPlug(Plug):
	node : Flow = None
	pass
class UDivisionsPlug(Plug):
	node : Flow = None
	pass
# endregion


# define node class
class Flow(_BASE_):
	xCoord_ : XCoordPlug = PlugDescriptor("xCoord")
	yCoord_ : YCoordPlug = PlugDescriptor("yCoord")
	zCoord_ : ZCoordPlug = PlugDescriptor("zCoord")
	allCoords_ : AllCoordsPlug = PlugDescriptor("allCoords")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	centerX_ : CenterXPlug = PlugDescriptor("centerX")
	centerY_ : CenterYPlug = PlugDescriptor("centerY")
	centerZ_ : CenterZPlug = PlugDescriptor("centerZ")
	center_ : CenterPlug = PlugDescriptor("center")
	curve_ : CurvePlug = PlugDescriptor("curve")
	defMatrixInv_ : DefMatrixInvPlug = PlugDescriptor("defMatrixInv")
	defPts_ : DefPtsPlug = PlugDescriptor("defPts")
	inBaseMatrix_ : InBaseMatrixPlug = PlugDescriptor("inBaseMatrix")
	latticeOnObject_ : LatticeOnObjectPlug = PlugDescriptor("latticeOnObject")
	motionPath_ : MotionPathPlug = PlugDescriptor("motionPath")
	objectWorldMatrix_ : ObjectWorldMatrixPlug = PlugDescriptor("objectWorldMatrix")
	orientMatrix_ : OrientMatrixPlug = PlugDescriptor("orientMatrix")
	outBaseMatrix_ : OutBaseMatrixPlug = PlugDescriptor("outBaseMatrix")
	parmValue_ : ParmValuePlug = PlugDescriptor("parmValue")
	sDivisions_ : SDivisionsPlug = PlugDescriptor("sDivisions")
	setFrontAxis_ : SetFrontAxisPlug = PlugDescriptor("setFrontAxis")
	setUpAxis_ : SetUpAxisPlug = PlugDescriptor("setUpAxis")
	tDivisions_ : TDivisionsPlug = PlugDescriptor("tDivisions")
	uDivisions_ : UDivisionsPlug = PlugDescriptor("uDivisions")

	# node attributes

	typeName = "flow"
	apiTypeInt = 72
	apiTypeStr = "kFlow"
	typeIdInt = 1179406167
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["xCoord", "yCoord", "zCoord", "allCoords", "binMembership", "centerX", "centerY", "centerZ", "center", "curve", "defMatrixInv", "defPts", "inBaseMatrix", "latticeOnObject", "motionPath", "objectWorldMatrix", "orientMatrix", "outBaseMatrix", "parmValue", "sDivisions", "setFrontAxis", "setUpAxis", "tDivisions", "uDivisions"]
	nodeLeafPlugs = ["allCoords", "binMembership", "center", "curve", "defMatrixInv", "defPts", "inBaseMatrix", "latticeOnObject", "motionPath", "objectWorldMatrix", "orientMatrix", "outBaseMatrix", "parmValue", "sDivisions", "setFrontAxis", "setUpAxis", "tDivisions", "uDivisions"]
	pass

