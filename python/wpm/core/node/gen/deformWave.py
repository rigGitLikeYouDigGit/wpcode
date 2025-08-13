

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
	node : DeformWave = None
	pass
class DropoffPlug(Plug):
	node : DeformWave = None
	pass
class DropoffPositionPlug(Plug):
	node : DeformWave = None
	pass
class MaxRadiusPlug(Plug):
	node : DeformWave = None
	pass
class MinRadiusPlug(Plug):
	node : DeformWave = None
	pass
class OffsetPlug(Plug):
	node : DeformWave = None
	pass
class WavelengthPlug(Plug):
	node : DeformWave = None
	pass
# endregion


# define node class
class DeformWave(DeformFunc):
	amplitude_ : AmplitudePlug = PlugDescriptor("amplitude")
	dropoff_ : DropoffPlug = PlugDescriptor("dropoff")
	dropoffPosition_ : DropoffPositionPlug = PlugDescriptor("dropoffPosition")
	maxRadius_ : MaxRadiusPlug = PlugDescriptor("maxRadius")
	minRadius_ : MinRadiusPlug = PlugDescriptor("minRadius")
	offset_ : OffsetPlug = PlugDescriptor("offset")
	wavelength_ : WavelengthPlug = PlugDescriptor("wavelength")

	# node attributes

	typeName = "deformWave"
	apiTypeInt = 630
	apiTypeStr = "kDeformWave"
	typeIdInt = 1178883926
	MFnCls = om.MFnDagNode
	pass

