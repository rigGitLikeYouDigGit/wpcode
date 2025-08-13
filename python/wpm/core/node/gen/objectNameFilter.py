

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
ObjectFilter = retriever.getNodeCls("ObjectFilter")
assert ObjectFilter
if T.TYPE_CHECKING:
	from .. import ObjectFilter

# add node doc



# region plug type defs
class AttrNamePlug(Plug):
	node : ObjectNameFilter = None
	pass
class NameStringsPlug(Plug):
	node : ObjectNameFilter = None
	pass
class RegExpPlug(Plug):
	node : ObjectNameFilter = None
	pass
# endregion


# define node class
class ObjectNameFilter(ObjectFilter):
	attrName_ : AttrNamePlug = PlugDescriptor("attrName")
	nameStrings_ : NameStringsPlug = PlugDescriptor("nameStrings")
	regExp_ : RegExpPlug = PlugDescriptor("regExp")

	# node attributes

	typeName = "objectNameFilter"
	apiTypeInt = 678
	apiTypeStr = "kObjectNameFilter"
	typeIdInt = 1330529868
	MFnCls = om.MFnDependencyNode
	pass

