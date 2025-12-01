

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
DeformFunc = retriever.getNodeCls("DeformFunc")
assert DeformFunc
if T.TYPE_CHECKING:
	from .. import DeformFunc

# add node doc



# region plug type defs
class AmplitudePlug(Plug):
	node : DeformSine = None
	pass
class DropoffPlug(Plug):
	node : DeformSine = None
	pass
class HighBoundPlug(Plug):
	node : DeformSine = None
	pass
class LowBoundPlug(Plug):
	node : DeformSine = None
	pass
class OffsetPlug(Plug):
	node : DeformSine = None
	pass
class WavelengthPlug(Plug):
	node : DeformSine = None
	pass
# endregion


# define node class
class DeformSine(DeformFunc):
	amplitude_ : AmplitudePlug = PlugDescriptor("amplitude")
	dropoff_ : DropoffPlug = PlugDescriptor("dropoff")
	highBound_ : HighBoundPlug = PlugDescriptor("highBound")
	lowBound_ : LowBoundPlug = PlugDescriptor("lowBound")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	wavelength_ : WavelengthPlug = PlugDescriptor("wavelength")

	# node attributes

	typeName = "deformSine"
	apiTypeInt = 629
	apiTypeStr = "kDeformSine"
	typeIdInt = 1178882894
	MFnCls = om.MFnDagNode
	nodeLeafClassAttrs = ["amplitude", "dropoff", "highBound", "lowBound", "offset", "wavelength"]
	nodeLeafPlugs = ["amplitude", "dropoff", "highBound", "lowBound", "offset", "wavelength"]
	pass

