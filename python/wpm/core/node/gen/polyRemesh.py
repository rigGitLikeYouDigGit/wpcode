

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	PolyModifierWorld = Catalogue.PolyModifierWorld
else:
	from .. import retriever
	PolyModifierWorld = retriever.getNodeCls("PolyModifierWorld")
	assert PolyModifierWorld

# add node doc



# region plug type defs
class CollapseThresholdPlug(Plug):
	node : PolyRemesh = None
	pass
class InterpolationTypePlug(Plug):
	node : PolyRemesh = None
	pass
class MaxEdgeLengthPlug(Plug):
	node : PolyRemesh = None
	pass
class MaxTriangleCountPlug(Plug):
	node : PolyRemesh = None
	pass
class SmoothStrengthPlug(Plug):
	node : PolyRemesh = None
	pass
class TessellateBordersPlug(Plug):
	node : PolyRemesh = None
	pass
# endregion


# define node class
class PolyRemesh(PolyModifierWorld):
	collapseThreshold_ : CollapseThresholdPlug = PlugDescriptor("collapseThreshold")
	interpolationType_ : InterpolationTypePlug = PlugDescriptor("interpolationType")
	maxEdgeLength_ : MaxEdgeLengthPlug = PlugDescriptor("maxEdgeLength")
	maxTriangleCount_ : MaxTriangleCountPlug = PlugDescriptor("maxTriangleCount")
	smoothStrength_ : SmoothStrengthPlug = PlugDescriptor("smoothStrength")
	tessellateBorders_ : TessellateBordersPlug = PlugDescriptor("tessellateBorders")

	# node attributes

	typeName = "polyRemesh"
	apiTypeInt = 1113
	apiTypeStr = "kPolyRemesh"
	typeIdInt = 1347571016
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["collapseThreshold", "interpolationType", "maxEdgeLength", "maxTriangleCount", "smoothStrength", "tessellateBorders"]
	nodeLeafPlugs = ["collapseThreshold", "interpolationType", "maxEdgeLength", "maxTriangleCount", "smoothStrength", "tessellateBorders"]
	pass

