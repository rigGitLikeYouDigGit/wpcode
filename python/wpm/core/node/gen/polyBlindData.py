

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
	node : PolyBlindData = None
	pass
class BlindDataEntriesAreNewPlug(Plug):
	node : PolyBlindData = None
	pass
class InMeshPlug(Plug):
	node : PolyBlindData = None
	pass
class OutMeshPlug(Plug):
	node : PolyBlindData = None
	pass
class TypeIdPlug(Plug):
	node : PolyBlindData = None
	pass
# endregion


# define node class
class PolyBlindData(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	blindDataEntriesAreNew_ : BlindDataEntriesAreNewPlug = PlugDescriptor("blindDataEntriesAreNew")
	inMesh_ : InMeshPlug = PlugDescriptor("inMesh")
	outMesh_ : OutMeshPlug = PlugDescriptor("outMesh")
	typeId_ : TypeIdPlug = PlugDescriptor("typeId")

	# node attributes

	typeName = "polyBlindData"
	apiTypeInt = 758
	apiTypeStr = "kPolyBlindData"
	typeIdInt = 1296188500
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["binMembership", "blindDataEntriesAreNew", "inMesh", "outMesh", "typeId"]
	nodeLeafPlugs = ["binMembership", "blindDataEntriesAreNew", "inMesh", "outMesh", "typeId"]
	pass

