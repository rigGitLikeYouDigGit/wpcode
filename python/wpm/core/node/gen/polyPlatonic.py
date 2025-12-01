

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
	node : PolyPlatonic = None
	pass
class HeightBaselinePlug(Plug):
	node : PolyPlatonic = None
	pass
class OutputPlug(Plug):
	node : PolyPlatonic = None
	pass
class PrimitivePlug(Plug):
	node : PolyPlatonic = None
	pass
class RadiusPlug(Plug):
	node : PolyPlatonic = None
	pass
class SphericalInflationPlug(Plug):
	node : PolyPlatonic = None
	pass
class SubdivisionModePlug(Plug):
	node : PolyPlatonic = None
	pass
class SubdivisionsPlug(Plug):
	node : PolyPlatonic = None
	pass
# endregion


# define node class
class PolyPlatonic(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	heightBaseline_ : HeightBaselinePlug = PlugDescriptor("heightBaseline")
	output_ : OutputPlug = PlugDescriptor("output")
	primitive_ : PrimitivePlug = PlugDescriptor("primitive")
	radius_ : RadiusPlug = PlugDescriptor("radius")
	sphericalInflation_ : SphericalInflationPlug = PlugDescriptor("sphericalInflation")
	subdivisionMode_ : SubdivisionModePlug = PlugDescriptor("subdivisionMode")
	subdivisions_ : SubdivisionsPlug = PlugDescriptor("subdivisions")

	# node attributes

	typeName = "polyPlatonic"
	typeIdInt = 1195049
	nodeLeafClassAttrs = ["binMembership", "heightBaseline", "output", "primitive", "radius", "sphericalInflation", "subdivisionMode", "subdivisions"]
	nodeLeafPlugs = ["binMembership", "heightBaseline", "output", "primitive", "radius", "sphericalInflation", "subdivisionMode", "subdivisions"]
	pass

