

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
PolyModifier = retriever.getNodeCls("PolyModifier")
assert PolyModifier
if T.TYPE_CHECKING:
	from .. import PolyModifier

# add node doc



# region plug type defs
class TweakXPlug(Plug):
	parent : TweakPlug = PlugDescriptor("tweak")
	node : PolyTweak = None
	pass
class TweakYPlug(Plug):
	parent : TweakPlug = PlugDescriptor("tweak")
	node : PolyTweak = None
	pass
class TweakZPlug(Plug):
	parent : TweakPlug = PlugDescriptor("tweak")
	node : PolyTweak = None
	pass
class TweakPlug(Plug):
	tweakX_ : TweakXPlug = PlugDescriptor("tweakX")
	tx_ : TweakXPlug = PlugDescriptor("tweakX")
	tweakY_ : TweakYPlug = PlugDescriptor("tweakY")
	ty_ : TweakYPlug = PlugDescriptor("tweakY")
	tweakZ_ : TweakZPlug = PlugDescriptor("tweakZ")
	tz_ : TweakZPlug = PlugDescriptor("tweakZ")
	node : PolyTweak = None
	pass
# endregion


# define node class
class PolyTweak(PolyModifier):
	tweakX_ : TweakXPlug = PlugDescriptor("tweakX")
	tweakY_ : TweakYPlug = PlugDescriptor("tweakY")
	tweakZ_ : TweakZPlug = PlugDescriptor("tweakZ")
	tweak_ : TweakPlug = PlugDescriptor("tweak")

	# node attributes

	typeName = "polyTweak"
	apiTypeInt = 402
	apiTypeStr = "kPolyTweak"
	typeIdInt = 1347704651
	MFnCls = om.MFnDependencyNode
	pass

