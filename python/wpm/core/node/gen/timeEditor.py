

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
class ActiveCompositionPlug(Plug):
	node : TimeEditor = None
	pass
class AnimationSourcePlug(Plug):
	parent : AttributesPlug = PlugDescriptor("attributes")
	node : TimeEditor = None
	pass
class AttributePlug(Plug):
	parent : AttributesPlug = PlugDescriptor("attributes")
	node : TimeEditor = None
	pass
class ValuePlug(Plug):
	parent : AttributesPlug = PlugDescriptor("attributes")
	node : TimeEditor = None
	pass
class AttributesPlug(Plug):
	animationSource_ : AnimationSourcePlug = PlugDescriptor("animationSource")
	as_ : AnimationSourcePlug = PlugDescriptor("animationSource")
	attribute_ : AttributePlug = PlugDescriptor("attribute")
	a_ : AttributePlug = PlugDescriptor("attribute")
	value_ : ValuePlug = PlugDescriptor("value")
	v_ : ValuePlug = PlugDescriptor("value")
	node : TimeEditor = None
	pass
class BinMembershipPlug(Plug):
	node : TimeEditor = None
	pass
class CompositionPlug(Plug):
	node : TimeEditor = None
	pass
class MutePlug(Plug):
	node : TimeEditor = None
	pass
class NextClipIdPlug(Plug):
	node : TimeEditor = None
	pass
# endregion


# define node class
class TimeEditor(_BASE_):
	activeComposition_ : ActiveCompositionPlug = PlugDescriptor("activeComposition")
	animationSource_ : AnimationSourcePlug = PlugDescriptor("animationSource")
	attribute_ : AttributePlug = PlugDescriptor("attribute")
	value_ : ValuePlug = PlugDescriptor("value")
	attributes_ : AttributesPlug = PlugDescriptor("attributes")
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	composition_ : CompositionPlug = PlugDescriptor("composition")
	mute_ : MutePlug = PlugDescriptor("mute")
	nextClipId_ : NextClipIdPlug = PlugDescriptor("nextClipId")

	# node attributes

	typeName = "timeEditor"
	apiTypeInt = 1106
	apiTypeStr = "kTimeEditor"
	typeIdInt = 1414350148
	MFnCls = om.MFnDependencyNode
	pass

