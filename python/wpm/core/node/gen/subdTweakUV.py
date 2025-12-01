

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
class UvTweakUPlug(Plug):
	parent : UvTweakPlug = PlugDescriptor("uvTweak")
	node : SubdTweakUV = None
	pass
class UvTweakVPlug(Plug):
	parent : UvTweakPlug = PlugDescriptor("uvTweak")
	node : SubdTweakUV = None
	pass
class UvTweakPlug(Plug):
	uvTweakU_ : UvTweakUPlug = PlugDescriptor("uvTweakU")
	tu_ : UvTweakUPlug = PlugDescriptor("uvTweakU")
	uvTweakV_ : UvTweakVPlug = PlugDescriptor("uvTweakV")
	tv_ : UvTweakVPlug = PlugDescriptor("uvTweakV")
	node : SubdTweakUV = None
	pass
# endregion


# define node class
class SubdTweakUV(SubdModifier):
	uvTweakU_ : UvTweakUPlug = PlugDescriptor("uvTweakU")
	uvTweakV_ : UvTweakVPlug = PlugDescriptor("uvTweakV")
	uvTweak_ : UvTweakPlug = PlugDescriptor("uvTweak")

	# node attributes

	typeName = "subdTweakUV"
	apiTypeInt = 871
	apiTypeStr = "kSubdTweakUV"
	typeIdInt = 1398035798
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["uvTweakU", "uvTweakV", "uvTweak"]
	nodeLeafPlugs = ["uvTweak"]
	pass

