

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyBase = retriever.getNodeCls("PolyBase")
assert PolyBase
if T.TYPE_CHECKING:
	from .. import PolyBase

# add node doc



# region plug type defs
class EndFacePlug(Plug):
	node : PolySeparate = None
	pass
class IcountPlug(Plug):
	node : PolySeparate = None
	pass
class InPlacePlug(Plug):
	node : PolySeparate = None
	pass
class InputPolyPlug(Plug):
	node : PolySeparate = None
	pass
class RemShellsPlug(Plug):
	node : PolySeparate = None
	pass
class StartFacePlug(Plug):
	node : PolySeparate = None
	pass
class UseOldPolyArchitecturePlug(Plug):
	node : PolySeparate = None
	pass
class UserSpecifiedShellsPlug(Plug):
	node : PolySeparate = None
	pass
# endregion


# define node class
class PolySeparate(PolyBase):
	endFace_ : EndFacePlug = PlugDescriptor("endFace")
	icount_ : IcountPlug = PlugDescriptor("icount")
	inPlace_ : InPlacePlug = PlugDescriptor("inPlace")
	inputPoly_ : InputPolyPlug = PlugDescriptor("inputPoly")
	remShells_ : RemShellsPlug = PlugDescriptor("remShells")
	startFace_ : StartFacePlug = PlugDescriptor("startFace")
	useOldPolyArchitecture_ : UseOldPolyArchitecturePlug = PlugDescriptor("useOldPolyArchitecture")
	userSpecifiedShells_ : UserSpecifiedShellsPlug = PlugDescriptor("userSpecifiedShells")

	# node attributes

	typeName = "polySeparate"
	apiTypeInt = 463
	apiTypeStr = "kPolySeparate"
	typeIdInt = 1347634512
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["endFace", "icount", "inPlace", "inputPoly", "remShells", "startFace", "useOldPolyArchitecture", "userSpecifiedShells"]
	nodeLeafPlugs = ["endFace", "icount", "inPlace", "inputPoly", "remShells", "startFace", "useOldPolyArchitecture", "userSpecifiedShells"]
	pass

