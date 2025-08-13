

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifierUV = retriever.getNodeCls("PolyModifierUV")
assert PolyModifierUV
if T.TYPE_CHECKING:
	from .. import PolyModifierUV

# add node doc



# region plug type defs
class ApplyToShellPlug(Plug):
	node : PolyOptUvs = None
	pass
class AreaWeightPlug(Plug):
	node : PolyOptUvs = None
	pass
class GlobalBlendPlug(Plug):
	node : PolyOptUvs = None
	pass
class GlobalMethodBlendPlug(Plug):
	node : PolyOptUvs = None
	pass
class IterationsPlug(Plug):
	node : PolyOptUvs = None
	pass
class OptimizeAxisPlug(Plug):
	node : PolyOptUvs = None
	pass
class PinSelectedPlug(Plug):
	node : PolyOptUvs = None
	pass
class PinUvBorderPlug(Plug):
	node : PolyOptUvs = None
	pass
class ScalePlug(Plug):
	node : PolyOptUvs = None
	pass
class StoppingThresholdPlug(Plug):
	node : PolyOptUvs = None
	pass
class UseScalePlug(Plug):
	node : PolyOptUvs = None
	pass
# endregion


# define node class
class PolyOptUvs(PolyModifierUV):
	applyToShell_ : ApplyToShellPlug = PlugDescriptor("applyToShell")
	areaWeight_ : AreaWeightPlug = PlugDescriptor("areaWeight")
	globalBlend_ : GlobalBlendPlug = PlugDescriptor("globalBlend")
	globalMethodBlend_ : GlobalMethodBlendPlug = PlugDescriptor("globalMethodBlend")
	iterations_ : IterationsPlug = PlugDescriptor("iterations")
	optimizeAxis_ : OptimizeAxisPlug = PlugDescriptor("optimizeAxis")
	pinSelected_ : PinSelectedPlug = PlugDescriptor("pinSelected")
	pinUvBorder_ : PinUvBorderPlug = PlugDescriptor("pinUvBorder")
	scale_ : ScalePlug = PlugDescriptor("scale")
	stoppingThreshold_ : StoppingThresholdPlug = PlugDescriptor("stoppingThreshold")
	useScale_ : UseScalePlug = PlugDescriptor("useScale")

	# node attributes

	typeName = "polyOptUvs"
	typeIdInt = 1347376470
	pass

