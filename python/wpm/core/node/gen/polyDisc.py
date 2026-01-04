

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
	node : PolyDisc = None
	pass
class HeightBaselinePlug(Plug):
	node : PolyDisc = None
	pass
class OutputPlug(Plug):
	node : PolyDisc = None
	pass
class RadiusPlug(Plug):
	node : PolyDisc = None
	pass
class SidesPlug(Plug):
	node : PolyDisc = None
	pass
class SubdivisionModePlug(Plug):
	node : PolyDisc = None
	pass
class SubdivisionsPlug(Plug):
	node : PolyDisc = None
	pass
# endregion


# define node class
class PolyDisc(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	heightBaseline_ : HeightBaselinePlug = PlugDescriptor("heightBaseline")
	output_ : OutputPlug = PlugDescriptor("output")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	sides_ : SidesPlug = PlugDescriptor("sides")
	subdivisionMode_ : SubdivisionModePlug = PlugDescriptor("subdivisionMode")
	subdivisions_ : SubdivisionsPlug = PlugDescriptor("subdivisions")

	# node attributes

	typeName = "polyDisc"
	typeIdInt = 1195050
	nodeLeafClassAttrs = ["binMembership", "heightBaseline", "output", "radius", "sides", "subdivisionMode", "subdivisions"]
	nodeLeafPlugs = ["binMembership", "heightBaseline", "output", "radius", "sides", "subdivisionMode", "subdivisions"]
	pass

