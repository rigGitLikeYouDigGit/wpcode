

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifierWorld = retriever.getNodeCls("PolyModifierWorld")
assert PolyModifierWorld
if T.TYPE_CHECKING:
	from .. import PolyModifierWorld

# add node doc



# region plug type defs
class AlignmentPlug(Plug):
	node : PolyCircularize = None
	pass
class DivisionsPlug(Plug):
	node : PolyCircularize = None
	pass
class EvenlyDistributePlug(Plug):
	node : PolyCircularize = None
	pass
class NormalOffsetPlug(Plug):
	node : PolyCircularize = None
	pass
class NormalOrientationPlug(Plug):
	node : PolyCircularize = None
	pass
class RadialOffsetPlug(Plug):
	node : PolyCircularize = None
	pass
class RelaxInteriorPlug(Plug):
	node : PolyCircularize = None
	pass
class SmoothingAnglePlug(Plug):
	node : PolyCircularize = None
	pass
class SupportingEdgesPlug(Plug):
	node : PolyCircularize = None
	pass
class TwistPlug(Plug):
	node : PolyCircularize = None
	pass
# endregion


# define node class
class PolyCircularize(PolyModifierWorld):
	alignment_ : AlignmentPlug = PlugDescriptor("alignment")
	divisions_ : DivisionsPlug = PlugDescriptor("divisions")
	evenlyDistribute_ : EvenlyDistributePlug = PlugDescriptor("evenlyDistribute")
	normalOffset_ : NormalOffsetPlug = PlugDescriptor("normalOffset")
	normalOrientation_ : NormalOrientationPlug = PlugDescriptor("normalOrientation")
	radialOffset_ : RadialOffsetPlug = PlugDescriptor("radialOffset")
	relaxInterior_ : RelaxInteriorPlug = PlugDescriptor("relaxInterior")
	smoothingAngle_ : SmoothingAnglePlug = PlugDescriptor("smoothingAngle")
	supportingEdges_ : SupportingEdgesPlug = PlugDescriptor("supportingEdges")
	twist_ : TwistPlug = PlugDescriptor("twist")

	# node attributes

	typeName = "polyCircularize"
	apiTypeInt = 1131
	apiTypeStr = "kPolyCircularize"
	typeIdInt = 1346589251
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["alignment", "divisions", "evenlyDistribute", "normalOffset", "normalOrientation", "radialOffset", "relaxInterior", "smoothingAngle", "supportingEdges", "twist"]
	nodeLeafPlugs = ["alignment", "divisions", "evenlyDistribute", "normalOffset", "normalOrientation", "radialOffset", "relaxInterior", "smoothingAngle", "supportingEdges", "twist"]
	pass

