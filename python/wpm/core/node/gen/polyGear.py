

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
_BASE_ = retriever.getNodeCls("_BASE_")
assert _BASE_
if T.TYPE_CHECKING:
	from .. import _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : PolyGear = None
	pass
class GearMiddlePlug(Plug):
	node : PolyGear = None
	pass
class GearOffsetPlug(Plug):
	node : PolyGear = None
	pass
class GearSpacingPlug(Plug):
	node : PolyGear = None
	pass
class GearTipPlug(Plug):
	node : PolyGear = None
	pass
class HeightPlug(Plug):
	node : PolyGear = None
	pass
class HeightBaselinePlug(Plug):
	node : PolyGear = None
	pass
class HeightDivisionsPlug(Plug):
	node : PolyGear = None
	pass
class InternalRadiusPlug(Plug):
	node : PolyGear = None
	pass
class OutputPlug(Plug):
	node : PolyGear = None
	pass
class RadiusPlug(Plug):
	node : PolyGear = None
	pass
class SidesPlug(Plug):
	node : PolyGear = None
	pass
class TaperPlug(Plug):
	node : PolyGear = None
	pass
class TwistPlug(Plug):
	node : PolyGear = None
	pass
# endregion


# define node class
class PolyGear(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	gearMiddle_ : GearMiddlePlug = PlugDescriptor("gearMiddle")
	gearOffset_ : GearOffsetPlug = PlugDescriptor("gearOffset")
	gearSpacing_ : GearSpacingPlug = PlugDescriptor("gearSpacing")
	gearTip_ : GearTipPlug = PlugDescriptor("gearTip")
	height_ : HeightPlug = PlugDescriptor("height")
	heightBaseline_ : HeightBaselinePlug = PlugDescriptor("heightBaseline")
	heightDivisions_ : HeightDivisionsPlug = PlugDescriptor("heightDivisions")
	internalRadius_ : InternalRadiusPlug = PlugDescriptor("internalRadius")
	output_ : OutputPlug = PlugDescriptor("output")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	sides_ : SidesPlug = PlugDescriptor("sides")
	taper_ : TaperPlug = PlugDescriptor("taper")
	twist_ : TwistPlug = PlugDescriptor("twist")

	# node attributes

	typeName = "polyGear"
	typeIdInt = 1195052
	nodeLeafClassAttrs = ["binMembership", "gearMiddle", "gearOffset", "gearSpacing", "gearTip", "height", "heightBaseline", "heightDivisions", "internalRadius", "output", "radius", "sides", "taper", "twist"]
	nodeLeafPlugs = ["binMembership", "gearMiddle", "gearOffset", "gearSpacing", "gearTip", "height", "heightBaseline", "heightDivisions", "internalRadius", "output", "radius", "sides", "taper", "twist"]
	pass

