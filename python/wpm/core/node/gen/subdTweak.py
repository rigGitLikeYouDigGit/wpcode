

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
SubdModifier = retriever.getNodeCls("SubdModifier")
assert SubdModifier
if T.TYPE_CHECKING:
	from .. import SubdModifier

# add node doc



# region plug type defs
class Map64BitIndicesPlug(Plug):
	node : SubdTweak = None
	pass
class TweakXPlug(Plug):
	parent : TweakPlug = PlugDescriptor("tweak")
	node : SubdTweak = None
	pass
class TweakYPlug(Plug):
	parent : TweakPlug = PlugDescriptor("tweak")
	node : SubdTweak = None
	pass
class TweakZPlug(Plug):
	parent : TweakPlug = PlugDescriptor("tweak")
	node : SubdTweak = None
	pass
class TweakPlug(Plug):
	tweakX_ : TweakXPlug = PlugDescriptor("tweakX")
	tx_ : TweakXPlug = PlugDescriptor("tweakX")
	tweakY_ : TweakYPlug = PlugDescriptor("tweakY")
	ty_ : TweakYPlug = PlugDescriptor("tweakY")
	tweakZ_ : TweakZPlug = PlugDescriptor("tweakZ")
	tz_ : TweakZPlug = PlugDescriptor("tweakZ")
	node : SubdTweak = None
	pass
# endregion


# define node class
class SubdTweak(SubdModifier):
	map64BitIndices_ : Map64BitIndicesPlug = PlugDescriptor("map64BitIndices")
	tweakX_ : TweakXPlug = PlugDescriptor("tweakX")
	tweakY_ : TweakYPlug = PlugDescriptor("tweakY")
	tweakZ_ : TweakZPlug = PlugDescriptor("tweakZ")
	tweak_ : TweakPlug = PlugDescriptor("tweak")

	# node attributes

	typeName = "subdTweak"
	apiTypeInt = 883
	apiTypeStr = "kSubdTweak"
	typeIdInt = 1398036299
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["map64BitIndices", "tweakX", "tweakY", "tweakZ", "tweak"]
	nodeLeafPlugs = ["map64BitIndices", "tweak"]
	pass

