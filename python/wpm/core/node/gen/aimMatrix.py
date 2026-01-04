

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
class BinMembershipPlug(Plug):
	node : AimMatrix = None
	pass
class EnablePlug(Plug):
	node : AimMatrix = None
	pass
class EnvelopePlug(Plug):
	node : AimMatrix = None
	pass
class InputMatrixPlug(Plug):
	node : AimMatrix = None
	pass
class OutputMatrixPlug(Plug):
	node : AimMatrix = None
	pass
class PrimaryInputAxisXPlug(Plug):
	parent : PrimaryInputAxisPlug = PlugDescriptor("primaryInputAxis")
	node : AimMatrix = None
	pass
class PrimaryInputAxisYPlug(Plug):
	parent : PrimaryInputAxisPlug = PlugDescriptor("primaryInputAxis")
	node : AimMatrix = None
	pass
class PrimaryInputAxisZPlug(Plug):
	parent : PrimaryInputAxisPlug = PlugDescriptor("primaryInputAxis")
	node : AimMatrix = None
	pass
class PrimaryInputAxisPlug(Plug):
	parent : PrimaryPlug = PlugDescriptor("primary")
	primaryInputAxisX_ : PrimaryInputAxisXPlug = PlugDescriptor("primaryInputAxisX")
	pmx_ : PrimaryInputAxisXPlug = PlugDescriptor("primaryInputAxisX")
	primaryInputAxisY_ : PrimaryInputAxisYPlug = PlugDescriptor("primaryInputAxisY")
	pmy_ : PrimaryInputAxisYPlug = PlugDescriptor("primaryInputAxisY")
	primaryInputAxisZ_ : PrimaryInputAxisZPlug = PlugDescriptor("primaryInputAxisZ")
	pmz_ : PrimaryInputAxisZPlug = PlugDescriptor("primaryInputAxisZ")
	node : AimMatrix = None
	pass
class PrimaryModePlug(Plug):
	parent : PrimaryPlug = PlugDescriptor("primary")
	node : AimMatrix = None
	pass
class PrimaryTargetMatrixPlug(Plug):
	parent : PrimaryPlug = PlugDescriptor("primary")
	node : AimMatrix = None
	pass
class PrimaryTargetVectorXPlug(Plug):
	parent : PrimaryTargetVectorPlug = PlugDescriptor("primaryTargetVector")
	node : AimMatrix = None
	pass
class PrimaryTargetVectorYPlug(Plug):
	parent : PrimaryTargetVectorPlug = PlugDescriptor("primaryTargetVector")
	node : AimMatrix = None
	pass
class PrimaryTargetVectorZPlug(Plug):
	parent : PrimaryTargetVectorPlug = PlugDescriptor("primaryTargetVector")
	node : AimMatrix = None
	pass
class PrimaryTargetVectorPlug(Plug):
	parent : PrimaryPlug = PlugDescriptor("primary")
	primaryTargetVectorX_ : PrimaryTargetVectorXPlug = PlugDescriptor("primaryTargetVectorX")
	pmvx_ : PrimaryTargetVectorXPlug = PlugDescriptor("primaryTargetVectorX")
	primaryTargetVectorY_ : PrimaryTargetVectorYPlug = PlugDescriptor("primaryTargetVectorY")
	pmvy_ : PrimaryTargetVectorYPlug = PlugDescriptor("primaryTargetVectorY")
	primaryTargetVectorZ_ : PrimaryTargetVectorZPlug = PlugDescriptor("primaryTargetVectorZ")
	pmvz_ : PrimaryTargetVectorZPlug = PlugDescriptor("primaryTargetVectorZ")
	node : AimMatrix = None
	pass
class PrimaryPlug(Plug):
	primaryInputAxis_ : PrimaryInputAxisPlug = PlugDescriptor("primaryInputAxis")
	pmi_ : PrimaryInputAxisPlug = PlugDescriptor("primaryInputAxis")
	primaryMode_ : PrimaryModePlug = PlugDescriptor("primaryMode")
	prmd_ : PrimaryModePlug = PlugDescriptor("primaryMode")
	primaryTargetMatrix_ : PrimaryTargetMatrixPlug = PlugDescriptor("primaryTargetMatrix")
	pmat_ : PrimaryTargetMatrixPlug = PlugDescriptor("primaryTargetMatrix")
	primaryTargetVector_ : PrimaryTargetVectorPlug = PlugDescriptor("primaryTargetVector")
	pmiv_ : PrimaryTargetVectorPlug = PlugDescriptor("primaryTargetVector")
	node : AimMatrix = None
	pass
class SecondaryInputAxisXPlug(Plug):
	parent : SecondaryInputAxisPlug = PlugDescriptor("secondaryInputAxis")
	node : AimMatrix = None
	pass
class SecondaryInputAxisYPlug(Plug):
	parent : SecondaryInputAxisPlug = PlugDescriptor("secondaryInputAxis")
	node : AimMatrix = None
	pass
class SecondaryInputAxisZPlug(Plug):
	parent : SecondaryInputAxisPlug = PlugDescriptor("secondaryInputAxis")
	node : AimMatrix = None
	pass
class SecondaryInputAxisPlug(Plug):
	parent : SecondaryPlug = PlugDescriptor("secondary")
	secondaryInputAxisX_ : SecondaryInputAxisXPlug = PlugDescriptor("secondaryInputAxisX")
	smx_ : SecondaryInputAxisXPlug = PlugDescriptor("secondaryInputAxisX")
	secondaryInputAxisY_ : SecondaryInputAxisYPlug = PlugDescriptor("secondaryInputAxisY")
	smy_ : SecondaryInputAxisYPlug = PlugDescriptor("secondaryInputAxisY")
	secondaryInputAxisZ_ : SecondaryInputAxisZPlug = PlugDescriptor("secondaryInputAxisZ")
	smz_ : SecondaryInputAxisZPlug = PlugDescriptor("secondaryInputAxisZ")
	node : AimMatrix = None
	pass
class SecondaryModePlug(Plug):
	parent : SecondaryPlug = PlugDescriptor("secondary")
	node : AimMatrix = None
	pass
class SecondaryTargetMatrixPlug(Plug):
	parent : SecondaryPlug = PlugDescriptor("secondary")
	node : AimMatrix = None
	pass
class SecondaryTargetVectorXPlug(Plug):
	parent : SecondaryTargetVectorPlug = PlugDescriptor("secondaryTargetVector")
	node : AimMatrix = None
	pass
class SecondaryTargetVectorYPlug(Plug):
	parent : SecondaryTargetVectorPlug = PlugDescriptor("secondaryTargetVector")
	node : AimMatrix = None
	pass
class SecondaryTargetVectorZPlug(Plug):
	parent : SecondaryTargetVectorPlug = PlugDescriptor("secondaryTargetVector")
	node : AimMatrix = None
	pass
class SecondaryTargetVectorPlug(Plug):
	parent : SecondaryPlug = PlugDescriptor("secondary")
	secondaryTargetVectorX_ : SecondaryTargetVectorXPlug = PlugDescriptor("secondaryTargetVectorX")
	smvx_ : SecondaryTargetVectorXPlug = PlugDescriptor("secondaryTargetVectorX")
	secondaryTargetVectorY_ : SecondaryTargetVectorYPlug = PlugDescriptor("secondaryTargetVectorY")
	smvy_ : SecondaryTargetVectorYPlug = PlugDescriptor("secondaryTargetVectorY")
	secondaryTargetVectorZ_ : SecondaryTargetVectorZPlug = PlugDescriptor("secondaryTargetVectorZ")
	smvz_ : SecondaryTargetVectorZPlug = PlugDescriptor("secondaryTargetVectorZ")
	node : AimMatrix = None
	pass
class SecondaryPlug(Plug):
	secondaryInputAxis_ : SecondaryInputAxisPlug = PlugDescriptor("secondaryInputAxis")
	smi_ : SecondaryInputAxisPlug = PlugDescriptor("secondaryInputAxis")
	secondaryMode_ : SecondaryModePlug = PlugDescriptor("secondaryMode")
	sm_ : SecondaryModePlug = PlugDescriptor("secondaryMode")
	secondaryTargetMatrix_ : SecondaryTargetMatrixPlug = PlugDescriptor("secondaryTargetMatrix")
	smat_ : SecondaryTargetMatrixPlug = PlugDescriptor("secondaryTargetMatrix")
	secondaryTargetVector_ : SecondaryTargetVectorPlug = PlugDescriptor("secondaryTargetVector")
	smiv_ : SecondaryTargetVectorPlug = PlugDescriptor("secondaryTargetVector")
	node : AimMatrix = None
	pass
# endregion


# define node class
class AimMatrix(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	enable_ : EnablePlug = PlugDescriptor("enable")
	envelope_ : EnvelopePlug = PlugDescriptor("envelope")
	inputMatrix_ : InputMatrixPlug = PlugDescriptor("inputMatrix")
	outputMatrix_ : OutputMatrixPlug = PlugDescriptor("outputMatrix")
	primaryInputAxisX_ : PrimaryInputAxisXPlug = PlugDescriptor("primaryInputAxisX")
	primaryInputAxisY_ : PrimaryInputAxisYPlug = PlugDescriptor("primaryInputAxisY")
	primaryInputAxisZ_ : PrimaryInputAxisZPlug = PlugDescriptor("primaryInputAxisZ")
	primaryInputAxis_ : PrimaryInputAxisPlug = PlugDescriptor("primaryInputAxis")
	primaryMode_ : PrimaryModePlug = PlugDescriptor("primaryMode")
	primaryTargetMatrix_ : PrimaryTargetMatrixPlug = PlugDescriptor("primaryTargetMatrix")
	primaryTargetVectorX_ : PrimaryTargetVectorXPlug = PlugDescriptor("primaryTargetVectorX")
	primaryTargetVectorY_ : PrimaryTargetVectorYPlug = PlugDescriptor("primaryTargetVectorY")
	primaryTargetVectorZ_ : PrimaryTargetVectorZPlug = PlugDescriptor("primaryTargetVectorZ")
	primaryTargetVector_ : PrimaryTargetVectorPlug = PlugDescriptor("primaryTargetVector")
	primary_ : PrimaryPlug = PlugDescriptor("primary")
	secondaryInputAxisX_ : SecondaryInputAxisXPlug = PlugDescriptor("secondaryInputAxisX")
	secondaryInputAxisY_ : SecondaryInputAxisYPlug = PlugDescriptor("secondaryInputAxisY")
	secondaryInputAxisZ_ : SecondaryInputAxisZPlug = PlugDescriptor("secondaryInputAxisZ")
	secondaryInputAxis_ : SecondaryInputAxisPlug = PlugDescriptor("secondaryInputAxis")
	secondaryMode_ : SecondaryModePlug = PlugDescriptor("secondaryMode")
	secondaryTargetMatrix_ : SecondaryTargetMatrixPlug = PlugDescriptor("secondaryTargetMatrix")
	secondaryTargetVectorX_ : SecondaryTargetVectorXPlug = PlugDescriptor("secondaryTargetVectorX")
	secondaryTargetVectorY_ : SecondaryTargetVectorYPlug = PlugDescriptor("secondaryTargetVectorY")
	secondaryTargetVectorZ_ : SecondaryTargetVectorZPlug = PlugDescriptor("secondaryTargetVectorZ")
	secondaryTargetVector_ : SecondaryTargetVectorPlug = PlugDescriptor("secondaryTargetVector")
	secondary_ : SecondaryPlug = PlugDescriptor("secondary")

	# node attributes

	typeName = "aimMatrix"
	apiTypeInt = 1139
	apiTypeStr = "kAimMatrix"
	typeIdInt = 1095582036
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "enable", "envelope", "inputMatrix", "outputMatrix", "primaryInputAxisX", "primaryInputAxisY", "primaryInputAxisZ", "primaryInputAxis", "primaryMode", "primaryTargetMatrix", "primaryTargetVectorX", "primaryTargetVectorY", "primaryTargetVectorZ", "primaryTargetVector", "primary", "secondaryInputAxisX", "secondaryInputAxisY", "secondaryInputAxisZ", "secondaryInputAxis", "secondaryMode", "secondaryTargetMatrix", "secondaryTargetVectorX", "secondaryTargetVectorY", "secondaryTargetVectorZ", "secondaryTargetVector", "secondary"]
	nodeLeafPlugs = ["binMembership", "enable", "envelope", "inputMatrix", "outputMatrix", "primary", "secondary"]
	pass

