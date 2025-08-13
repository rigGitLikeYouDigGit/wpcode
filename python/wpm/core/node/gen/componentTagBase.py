

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
	node : ComponentTagBase = None
	pass
class ComponentTagContentsPlug(Plug):
	parent : ComponentTagsPlug = PlugDescriptor("componentTags")
	node : ComponentTagBase = None
	pass
class ComponentTagNamePlug(Plug):
	parent : ComponentTagsPlug = PlugDescriptor("componentTags")
	node : ComponentTagBase = None
	pass
class ComponentTagsPlug(Plug):
	componentTagContents_ : ComponentTagContentsPlug = PlugDescriptor("componentTagContents")
	gtagcmp_ : ComponentTagContentsPlug = PlugDescriptor("componentTagContents")
	componentTagName_ : ComponentTagNamePlug = PlugDescriptor("componentTagName")
	gtagnm_ : ComponentTagNamePlug = PlugDescriptor("componentTagName")
	node : ComponentTagBase = None
	pass
class InputGeometryPlug(Plug):
	node : ComponentTagBase = None
	pass
class OutputGeometryPlug(Plug):
	node : ComponentTagBase = None
	pass
# endregion


# define node class
class ComponentTagBase(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	componentTagContents_ : ComponentTagContentsPlug = PlugDescriptor("componentTagContents")
	componentTagName_ : ComponentTagNamePlug = PlugDescriptor("componentTagName")
	componentTags_ : ComponentTagsPlug = PlugDescriptor("componentTags")
	inputGeometry_ : InputGeometryPlug = PlugDescriptor("inputGeometry")
	outputGeometry_ : OutputGeometryPlug = PlugDescriptor("outputGeometry")

	# node attributes

	typeName = "componentTagBase"
	typeIdInt = 1129595475
	pass

